# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
import logging
from tkinter import Variable
import numpy as np
import datetime

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss

from modules.until_module import PreTrainedModel, LayerNorm, CrossEn, MILNCELoss, MaxMarginRankingLoss
from modules.module_bert import BertModel, BertConfig, BertOnlyMLMHead
from modules.module_visual import VisualModel, VisualConfig, VisualOnlyMLMHead
from modules.module_cross import CrossModel, CrossConfig
from modules.module_decoder import DecoderModel, DecoderConfig
from torch.autograd import Variable
logger = logging.getLogger(__name__)


class UniVLPreTrainedModel(PreTrainedModel, nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def __init__(self, bert_config, visual_config, cross_config, decoder_config, *inputs, **kwargs):
        # utilize bert config as base config
        super(UniVLPreTrainedModel, self).__init__(bert_config)
        self.bert_config = bert_config
        self.visual_config = visual_config
        self.cross_config = cross_config
        self.decoder_config = decoder_config

        self.bert = None
        self.visual = None
        self.cross = None
        self.decoder = None
        
        

    @classmethod
    def from_pretrained(cls, pretrained_bert_name, visual_model_name, cross_model_name, decoder_model_name,
                        state_dict=None, cache_dir=None, type_vocab_size=2, *inputs, **kwargs):

        task_config = None
        if "task_config" in kwargs.keys():
            task_config = kwargs["task_config"]
            if not hasattr(task_config, "local_rank"):
                task_config.__dict__["local_rank"] = 0
            elif task_config.local_rank == -1:
                task_config.local_rank = 0

        bert_config, state_dict = BertConfig.get_config(pretrained_bert_name, cache_dir, type_vocab_size, state_dict, task_config=task_config)
        visual_config, _ = VisualConfig.get_config(visual_model_name, cache_dir, type_vocab_size, state_dict=None, task_config=task_config)
        cross_config, _ = CrossConfig.get_config(cross_model_name, cache_dir, type_vocab_size, state_dict=None, task_config=task_config)
        decoder_config, _ = DecoderConfig.get_config(decoder_model_name, cache_dir, type_vocab_size, state_dict=None, task_config=task_config)

        model = cls(bert_config, visual_config, cross_config, decoder_config, *inputs, **kwargs)
        
        assert model.bert is not None
        assert model.visual is not None

        if state_dict is not None:
            model = cls.init_preweight(model, state_dict, task_config=task_config)

        return model

class NormalizeVideo(nn.Module):
    def __init__(self, task_config):
        super(NormalizeVideo, self).__init__()
        self.visual_norm2d = LayerNorm(task_config.video_dim)

    def forward(self, video):
        video = torch.as_tensor(video).float()
        video = video.view(-1, video.shape[-2], video.shape[-1])
        video = self.visual_norm2d(video)
        return video

def show_log(task_config, info):
    if task_config is None or task_config.local_rank == 0:
        logger.warning(info)

def update_attr(target_name, target_config, target_attr_name, source_config, source_attr_name, default_value=None):
    if hasattr(source_config, source_attr_name):
        if default_value is None or getattr(source_config, source_attr_name) != default_value:
            setattr(target_config, target_attr_name, getattr(source_config, source_attr_name))
            show_log(source_config, "Set {}.{}: {}.".format(target_name,
                                                            target_attr_name, getattr(target_config, target_attr_name)))
    return target_config

def check_attr(target_name, task_config):
    return hasattr(task_config, target_name) and task_config.__dict__[target_name]

class UniVL(UniVLPreTrainedModel):
    def __init__(self, bert_config, visual_config, cross_config, decoder_config, task_config):
        super(UniVL, self).__init__(bert_config, visual_config, cross_config, decoder_config)
        self.task_config = task_config
        self.ignore_video_index = -1

        assert self.task_config.max_words <= bert_config.max_position_embeddings
        assert self.task_config.max_words <= decoder_config.max_target_embeddings
        assert self.task_config.max_frames <= visual_config.max_position_embeddings
        assert self.task_config.max_words + self.task_config.max_frames <= cross_config.max_position_embeddings

        self._stage_one = True
        self._stage_two = False

        if check_attr('stage_two', self.task_config):
            self._stage_one = False
            self._stage_two = self.task_config.stage_two
        show_log(task_config, "Stage-One:{}, Stage-Two:{}".format(self._stage_one, self._stage_two))

        self.train_sim_after_cross = False
        if self._stage_one and check_attr('train_sim_after_cross', self.task_config):
            self.train_sim_after_cross = True
            show_log(task_config, "Test retrieval after cross encoder.")

        # Text Encoder ===>
        bert_config = update_attr("bert_config", bert_config, "num_hidden_layers",
                                   self.task_config, "text_num_hidden_layers")
        self.bert = BertModel(bert_config)
        
        
        bert_word_embeddings_weight = self.bert.embeddings.word_embeddings.weight
        bert_position_embeddings_weight = self.bert.embeddings.position_embeddings.weight
        # <=== End of Text Encoder

        # Video Encoder ===>
        visual_config = update_attr("visual_config", visual_config, "num_hidden_layers",
                                    self.task_config, "visual_num_hidden_layers")
        self.visual = VisualModel(visual_config) # Transformer_based module
        visual_word_embeddings_weight = self.visual.embeddings.word_embeddings.weight
        # <=== End of Video Encoder

        if self._stage_one is False or self.train_sim_after_cross:
            # Cross Encoder ===>
            cross_config = update_attr("cross_config", cross_config, "num_hidden_layers",
                                        self.task_config, "cross_num_hidden_layers")
            self.cross = CrossModel(cross_config)
            # <=== End of Cross Encoder

            if self.train_sim_after_cross is False:
                # Decoder ===>
                decoder_config = update_attr("decoder_config", decoder_config, "num_decoder_layers",
                                           self.task_config, "decoder_num_hidden_layers")
                self.decoder = DecoderModel(decoder_config, bert_word_embeddings_weight, bert_position_embeddings_weight)
                # <=== End of Decoder

            if self.task_config.do_pretrain:
                self.cls = BertOnlyMLMHead(bert_config, bert_word_embeddings_weight)
                self.cls_visual = VisualOnlyMLMHead(visual_config, visual_word_embeddings_weight)
                self.alm_loss_fct = CrossEntropyLoss(ignore_index=-1)
                
            self.similarity_dense = nn.Linear(bert_config.hidden_size, 1)
            self.decoder_loss_fct = CrossEntropyLoss(ignore_index=-1)

        self.normalize_video = NormalizeVideo(task_config)
        self.weighted_m1 = Variable(torch.ones(task_config.batch_size// task_config.n_gpu,1,8),requires_grad = True)
        self.weighted_m2 = Variable(torch.ones(task_config.batch_size// task_config.n_gpu,1,8),requires_grad = True)
        mILNCELoss = MILNCELoss(batch_size=task_config.batch_size // task_config.n_gpu, n_pair=task_config.n_pair, )
        maxMarginRankingLoss = MaxMarginRankingLoss(margin=task_config.margin,
                                                    negative_weighting=task_config.negative_weighting,
                                                    batch_size=task_config.batch_size // task_config.n_gpu,
                                                    n_pair=task_config.n_pair,
                                                    hard_negative_rate=task_config.hard_negative_rate, )

        if task_config.use_mil:
            self.loss_fct = CrossEn() if self._stage_two else mILNCELoss
            self._pretrain_sim_loss_fct = mILNCELoss
        else:
            self.loss_fct = CrossEn() if self._stage_two else maxMarginRankingLoss
            self._pretrain_sim_loss_fct = maxMarginRankingLoss

        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids, attention_mask, video, siamese_video,video_mask=None,
                pairs_masked_text=None, pairs_token_labels=None, masked_video=None, video_labels_index=None,
                input_caption_ids=None, decoder_mask=None, output_caption_ids=None,siamese_trigger = False):
        starttime = time.perf_counter()
        
        input_ids = input_ids.view(-1, input_ids.shape[-1])
        token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
        attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
        video_mask = video_mask.view(-1, video_mask.shape[-1])
        # for _idx in range(siamese_clip_num):
        #     siamese_video[_idx] = self.normalize_video(siamese_video[_idx])
        video = self.normalize_video(video)
        # print(f'nomalized:{time.perf_counter() - starttime:.8f}s')
        
        starttime = time.perf_counter()
        if input_caption_ids is not None:
            input_caption_ids = input_caption_ids.view(-1, input_caption_ids.shape[-1])
            decoder_mask = decoder_mask.view(-1, decoder_mask.shape[-1])
        # visual_output = [siamese_num , bz , frm, hidden_layer]
        sequence_output, visual_output = self.get_sequence_visual_output(input_ids, token_type_ids, attention_mask,
                                                                         siamese_video,video, video_mask, siamese_trigger,shaped=True)
        # print(f'encoding:{time.perf_counter() - starttime:.8f}s')
        
        starttime = time.perf_counter()

        if self.training:
            loss = 0.
            if self._stage_one:
                sim_matrix = self.get_similarity_logits(sequence_output, visual_output, attention_mask,
                                                        video_mask, shaped=True)
                sim_loss = self.loss_fct(sim_matrix)
                loss += sim_loss

            if self._stage_two:
                if self.task_config.do_pretrain:
                    pairs_masked_text = pairs_masked_text.view(-1, pairs_masked_text.shape[-1])
                    pairs_token_labels = pairs_token_labels.view(-1, pairs_token_labels.shape[-1])

                    masked_video = self.normalize_video(masked_video)
                    video_labels_index = video_labels_index.view(-1, video_labels_index.shape[-1])

                    sequence_output_alm, visual_output_alm = self.get_sequence_visual_output(pairs_masked_text, token_type_ids,
                                                                                             attention_mask, masked_video, video_mask, shaped=True)

                    cross_output, pooled_output, concat_mask = self._get_cross_output(sequence_output_alm, visual_output_alm, attention_mask, video_mask)
                    sequence_cross_output, visual_cross_output = torch.split(cross_output, [attention_mask.size(-1), video_mask.size(-1)], dim=1)

                    alm_loss = self._calculate_mlm_loss(sequence_cross_output, pairs_token_labels)
                    loss += alm_loss

                    nce_loss = self._calculate_mfm_loss(visual_cross_output, video, video_mask, video_labels_index)
                    loss += nce_loss

                    sim_matrix = self.get_similarity_logits(sequence_output, visual_output, attention_mask, video_mask,
                                                            shaped=True, _pretrain_joint=True)
                    sim_loss_joint = self._pretrain_sim_loss_fct(sim_matrix)
                    loss += sim_loss_joint

                if (input_caption_ids is not None) and \
                        (self.task_config.do_pretrain
                         or (self.task_config.do_pretrain is False and self.task_config.task_type == "caption")):
                    if self.task_config.do_pretrain:
                        decoder_scores, res_tuples = self._get_decoder_score(sequence_output_alm, visual_output_alm,
                                                                             input_ids, attention_mask, video_mask,
                                                                             input_caption_ids, decoder_mask, shaped=True)
                    elif self.task_config.task_type == "caption":
                        if siamese_trigger == False:
                            
                            decoder_scores,res_tuples = self._get_decoder_score(sequence_output, visual_output,
                                                                             input_ids, attention_mask, video_mask,
                                                                             input_caption_ids, decoder_mask, shaped=True)
                        else:
                            # decoder_scores_siamese = [clips , bz , frms , hidden]
                            # _get_decoder_score完成其最原始的功能，能不改就不改
                            # print("siamese is using!")
                            # decoder_scores,res_tuples = self._get_decoder_score(sequence_output, visual_output[0],
                            #                                                  input_ids, attention_mask, video_mask,
                            #                                                  input_caption_ids, decoder_mask, shaped=True)
                            # decoder_scores_siamese = torch.zeros(siamese_clip_num,decoder_scores.size(0),decoder_scores.size(1),decoder_scores.size(2))
                            # decoder_scores_siamese[0] = decoder_scores
                            # for _idx in range(1,siamese_clip_num):
                            #     decoder_scores_temp,res_tuples = self._get_decoder_score(sequence_output, visual_output[_idx],
                            #                                                  input_ids, attention_mask, video_mask,
                            #                                                  input_caption_ids, decoder_mask, shaped=True)
                            #     decoder_scores_siamese[_idx] = decoder_scores_temp
                            video_mask = torch.ones(video_mask.size(0),5).to(video_mask.device)
                            decoder_scores_siamese,res_tuples = self._get_decoder_score(sequence_output, visual_output.reshape(-1,visual_output.size(-2),visual_output.size(-1)),
                                                                             input_ids, attention_mask, video_mask,
                                                                             input_caption_ids, decoder_mask, shaped=True)
                            # decoder_scores_siamese = [clip_num , bz , frm_max_length , score_features]
                            decoder_scores_siamese = decoder_scores_siamese.reshape(8,16,decoder_scores_siamese.size(-2),decoder_scores_siamese.size(-1))
                            decoder_scores = decoder_scores_siamese[0]
                        
                           
                            
                    else:
                        raise NotImplementedError
                    # visual_output = [clipNum , bz , 48 , 1024]
                    output_caption_ids = output_caption_ids.view(-1, output_caption_ids.shape[-1])
                    decoder_loss = self.decoder_loss_fct(decoder_scores.view(-1, self.bert_config.vocab_size), output_caption_ids.view(-1))
                    siamese_loss = 0.0
                    loss = decoder_loss
                    if siamese_trigger ==True:
                        siamese_similarity_m = self.get_crossPair_similarity_logits(sequence_output , visual_output,5).to(sequence_output.device)
                        
                        # siamese_similarity_m = [bz , 8 , 8 ]
                        
                        # decoder_scores_siamese = [8,16,48,30522]
                        # decoder_scores_siamese_T = [16,8,48,30522]
                        decoder_scores_siamese_T = decoder_scores_siamese.permute(1,0,2,3)
                        
                        dec_soft_label = torch.bmm(siamese_similarity_m,decoder_scores_siamese_T.view(decoder_scores_siamese_T.size(0),decoder_scores_siamese_T.size(1),-1))
                        dec_soft_label = torch.reshape(dec_soft_label,decoder_scores_siamese_T.size())
                        # dec_soft_label = [bz , clip_nums , max_length , 30522]
                        # weighted = W1 * A * P = [bz,48,30522]
                        self.weighted_m1 = self.weighted_m1.to(sequence_output.device)
                        self.weighted_m2 = self.weighted_m1.to(sequence_output.device)
                        
                        
                        dec_soft_label_weighted = torch.bmm(self.weighted_m1,dec_soft_label.view(dec_soft_label.size(0),dec_soft_label.size(1),-1))
                        dec_soft_label_weighted = torch.reshape(dec_soft_label_weighted,decoder_scores_siamese[0].size())
    
                        anchor_score_weighted = torch.bmm(self.weighted_m2,decoder_scores_siamese_T.view(decoder_scores_siamese_T.size(0),decoder_scores_siamese_T.size(1),-1))
                        anchor_score_weighted = torch.reshape(anchor_score_weighted,decoder_scores_siamese[0].size())
                    
                        final_score_pseudo = 0.6 * dec_soft_label_weighted + 0.4 *anchor_score_weighted
                        # GTH_pseudo = [16,48,30522]
                        anchor_score_raw = decoder_scores_siamese[0]
                        # Index = [16,48]
                        GTH_index = anchor_score_raw.argmax(-1)
                        # anchor_score_GTH = anchor_score_raw.gather(-1,GTH_index.unsqueeze(-1)).squeeze(-1)
                        
                        # 用learning from inside的方法，和anchor算loss
                        # siamese_loss = self.decoder_loss_fct(final_score_pseudo.view(-1,self.bert_config.vocab_size),GTH_index.view(-1))
                        
                        # 和GTH算loss
                        siamese_loss = self.decoder_loss_fct(final_score_pseudo.view(-1,self.bert_config.vocab_size).to(output_caption_ids.device),output_caption_ids.view(-1))
                        loss =loss +siamese_loss
                        # loss =loss
                        # print(f'decoding:{time.perf_counter() - starttime:.8f}s')
                        
                        
                        
                        
                        
                            
                            
                        
                    
                    
                    
                    
                    
                    
                    # decoder_loss_pseudo = self.decoder_loss_fct(decoder_scores_sia.view(-1, self.bert_config.vocab_size), output_caption_ids.view(-1))
                    
                    # decoder_loss_sia = decoder_loss_pseudo
                    
                    # loss += 0.7 *decoder_loss + 0.3 * decoder_loss_sia
                    

                if self.task_config.do_pretrain or self.task_config.task_type == "retrieval":
                    if self.task_config.do_pretrain:
                        sim_matrix_text_visual = self.get_similarity_logits(sequence_output_alm, visual_output_alm,
                                                                            attention_mask, video_mask, shaped=True)
                    elif self.task_config.task_type == "retrieval":
                        sim_matrix_text_visual = self.get_similarity_logits(sequence_output, visual_output,
                                                                            attention_mask, video_mask, shaped=True)
                    else:
                        raise NotImplementedError

                    sim_loss_text_visual = self.loss_fct(sim_matrix_text_visual)
                    loss += sim_loss_text_visual

            return loss
        else:
            return None

    def _calculate_mlm_loss(self, sequence_output_alm, pairs_token_labels):
        alm_scores = self.cls(sequence_output_alm)
        alm_loss = self.alm_loss_fct(alm_scores.view(-1, self.bert_config.vocab_size), pairs_token_labels.view(-1))
        return alm_loss

    def _calculate_mfm_loss(self, visual_output_alm, video, video_mask, video_labels_index):
        afm_scores = self.cls_visual(visual_output_alm)
        afm_scores_tr = afm_scores.view(-1, afm_scores.shape[-1])

        video_tr = video.permute(2, 0, 1)
        video_tr = video_tr.view(video_tr.shape[0], -1)

        logits_matrix = torch.mm(afm_scores_tr, video_tr)
        video_mask_float = video_mask.to(dtype=torch.float)
        mask_matrix = torch.mm(video_mask_float.view(-1, 1), video_mask_float.view(1, -1))
        masked_logits = logits_matrix + (1. - mask_matrix) * -1e8

        logpt = F.log_softmax(masked_logits, dim=-1)
        logpt = torch.diag(logpt)
        nce_loss = -logpt

        video_labels_index_mask = (video_labels_index != self.ignore_video_index)
        nce_loss = nce_loss.masked_select(video_labels_index_mask.view(-1))
        nce_loss = nce_loss.mean()
        return nce_loss
    def getSiameseClips(self,video,max_frames):
        siamese_output = torch.zeros(video.size(0),8,5,video.size(-1))
        sia_idx = np.zeros((8,5),int)
        single_ran_idx = np.zeros(5,int)
        for j in range(5):
            if j ==0:
                single_ran_idx[j] = np.random.randint(0,5*j+5)
            elif j<=2:
                single_ran_idx[j] = np.random.randint(single_ran_idx[j-1]+1,5*j+5)
            else:
                single_ran_idx[j] = np.random.randint(single_ran_idx[j-1]+1,max_frames-1-8-5+j)
        sia_idx[0]=single_ran_idx
        for i in range(1,8):
            sia_idx[i] = i+sia_idx[0]
        sia_idx = torch.from_numpy(sia_idx)
        # For Indexing
        sia_idx = torch.LongTensor(sia_idx).to(video.device)
        for batch_num in range(video.size(0)):
        
            single_batch_video = video[batch_num]
            # gathered_batch_video = [clip_num , 5 , 1024]
            gathered_batch_video = single_batch_video.unsqueeze(1).expand(48,5,768).gather(dim = 0 ,index=sia_idx.unsqueeze(2).expand(8,5,768))
            # gathered_batch_video = [clip_num, 5 + padding=48 , 1024] == [ 8 , 48 , 1024 ] 
            # need to padding second dim to match frm_max_length
            # gathered_batch_video_padding = padding_video(gathered_batch_video,max_frames)
            # tempoutput = [bz , 8 , 48 , 1024]
            siamese_output[batch_num] = gathered_batch_video
    
    
        siamese_video = siamese_output.permute(1,0,2,3)
        # output = [8 , bz , 48 , 1024]
        return siamese_video
    def get_sequence_visual_output(self, input_ids, token_type_ids, attention_mask, video,raw_video,video_mask, siamese_trigger,shaped=False):

        shaped_video = video.reshape(-1,video.size(-2),video.size(-1))
        
        siamese_video = torch.zeros(raw_video.size()).repeat(8,1,1)
        clip_num = 8
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            video = self.normalize_video(video)

        encoded_layers, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=True) # 里面一共有12层，取最后一层的输出即为Sequence output
        sequence_output = encoded_layers[-1]
        
        if siamese_trigger == False:
            visual_layers, _ = self.visual(raw_video, video_mask, output_all_encoded_layers=True)
            visual_output = visual_layers[-1]
            return sequence_output,visual_output
        # video = [clip*bz , 5,feature]
        frame_num = video.size(-2)
        video_mask = torch.ones(video.size(0),frame_num).long().to(sequence_output.device)
        video = video.to(sequence_output.device)

        visual_layers, _ = self.visual(video, video_mask, output_all_encoded_layers=True)
        visual_output = visual_layers[-1]
        
        # siamese_video = self.getSiameseClips(visual_output,48)
        siamese_video = visual_output.reshape(clip_num,visual_output.size(0) // clip_num,visual_output.size(-2),visual_output.size(-1))
        
        
        
        # video_mask = video_mask.repeat(clip_num,1)
        # visual_layers, _ = self.visual(shaped_video, video_mask, output_all_encoded_layers=True)
        # visual_output = visual_layers[-1]
        # visual_output = visual_output.reshape(clip_num , (visual_output.size(0) // clip_num) , visual_output.size(-2) , visual_output.size(-1))

        
        return sequence_output, siamese_video
    def get_sequence_visual_output_eval(self, input_ids, token_type_ids, attention_mask, video, video_mask, shaped=False):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            video = self.normalize_video(video)

        encoded_layers, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=True)
        sequence_output = encoded_layers[-1]

        visual_layers, _ = self.visual(video, video_mask, output_all_encoded_layers=True)
        visual_output = visual_layers[-1]

        return sequence_output, visual_output
    def _get_cross_output(self, sequence_output, visual_output, attention_mask, video_mask):
        # 涉及到visual_output第一维度的变化，所以要对sequence_output进行一个重复
        clip_num = visual_output.size(0) // sequence_output.size(0)
        sequence_output = sequence_output.repeat(clip_num,1,1)
        concat_features = torch.cat((sequence_output, visual_output), dim=1)  # concatnate tokens and frames
        concat_mask = torch.cat((attention_mask, video_mask), dim=1)
        concat_mask = concat_mask.repeat(clip_num,1)
        
        text_type_ = torch.zeros_like(attention_mask)
        video_type_ = torch.ones_like(video_mask)
        concat_type = torch.cat((text_type_, video_type_), dim=1)
        
        concat_type = concat_type.repeat(clip_num,1)
        concat_type = concat_type.long()
        cross_layers, pooled_output = self.cross(concat_features, concat_type, concat_mask, output_all_encoded_layers=True)
        cross_output = cross_layers[-1]

        return cross_output, pooled_output, concat_mask

    def _mean_pooling_for_similarity(self, sequence_output, visual_output, attention_mask, video_mask,):
        attention_mask_un = attention_mask.to(dtype=torch.float).unsqueeze(-1)
        attention_mask_un[:, 0, :] = 0.
        sequence_output = sequence_output * attention_mask_un
        text_out = torch.sum(sequence_output, dim=1) / torch.sum(attention_mask_un, dim=1, dtype=torch.float)

        video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
        visual_output = visual_output * video_mask_un
        video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)
        video_mask_un_sum[video_mask_un_sum == 0.] = 1.
        video_out = torch.sum(visual_output, dim=1) / video_mask_un_sum

        return text_out, video_out

    def _cross_similarity(self, sequence_output, visual_output, attention_mask, video_mask):
        b_text, s_text, h_text = sequence_output.size()
        b_visual, s_visual, h_visual = visual_output.size()

        retrieve_logits_list = []
        step_size = 5

        split_size = [step_size] * (b_text // step_size)
        release_size = b_text - sum(split_size)
        if release_size > 0:
            split_size += [release_size]

        sequence_output_splits = torch.split(sequence_output, split_size, dim=0)
        attention_mask_splits = torch.split(attention_mask, split_size, dim=0)
        for i in range(len(split_size)):
            sequence_output_row = sequence_output_splits[i]
            attention_mask_row = attention_mask_splits[i]
            sequence_output_l = sequence_output_row.unsqueeze(1).repeat(1, b_visual, 1, 1)
            sequence_output_l = sequence_output_l.view(-1, s_text, h_text)
            attention_mask_l = attention_mask_row.unsqueeze(1).repeat(1, b_visual, 1)
            attention_mask_l = attention_mask_l.view(-1, s_text)

            step_truth = sequence_output_row.size(0)
            visual_output_r = visual_output.unsqueeze(0).repeat(step_truth, 1, 1, 1)
            visual_output_r = visual_output_r.view(-1, s_visual, h_visual)
            video_mask_r = video_mask.unsqueeze(0).repeat(step_truth, 1, 1)
            video_mask_r = video_mask_r.view(-1, s_visual)

            cross_output, pooled_output, concat_mask = \
                self._get_cross_output(sequence_output_l, visual_output_r, attention_mask_l, video_mask_r)
            retrieve_logits_row = self.similarity_dense(pooled_output).squeeze(-1).view(step_truth, b_visual)

            retrieve_logits_list.append(retrieve_logits_row)
        retrieve_logits = torch.cat(retrieve_logits_list, dim=0)
        return retrieve_logits

    def get_similarity_logits(self, sequence_output, visual_output, attention_mask, video_mask, shaped=False, _pretrain_joint=False):
        if shaped is False:
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

        if (self._stage_two and _pretrain_joint is False) or self.train_sim_after_cross:
            retrieve_logits = self._cross_similarity(sequence_output, visual_output, attention_mask, video_mask)
        else:
            text_out, video_out = self._mean_pooling_for_similarity(sequence_output, visual_output, attention_mask, video_mask)
            if self.task_config.use_mil is False:
                text_out = F.normalize(text_out, dim=-1)
                video_out = F.normalize(video_out, dim=-1)
            retrieve_logits = torch.matmul(text_out, video_out.t())

        return retrieve_logits
    def get_crossPair_similarity_logits(self, sequence_output, visual_output , max_frm_length = 48):
        # visual_output = [clipNum , bz , 48 , 1024]
        clip_num = visual_output.size(0)
        batch_size = sequence_output.size(0)
        sequence_output = sequence_output.repeat(clip_num,1,1)
        visual_output = visual_output.reshape(-1,visual_output.size(-2),visual_output.size(-1))
        
        max_frm_length = visual_output.size(2)


        similarity_matrix = torch.zeros(batch_size,clip_num,clip_num)
        # concat_feature = torch.zeros(clip_num,batch_size,visual_output.size(2)*2,visual_output.size(3))
        # for _idx in range(batch_size):
        #     single_silimarity[_idx][_idx]=0.0
        visual_output = visual_output.to(sequence_output.device)
        
        # for i in range(clip_num):
        #     concat_feature[i] = torch.cat((sequence_output, visual_output[i]), dim=1)  # concatnate tokens and frames
        # concat_feature = [clip_num * bz , 53 , 768]
        concat_feature = torch.cat((sequence_output,visual_output),dim = 1)  
        concat_feature = concat_feature.reshape(batch_size,clip_num,-1)
        # concat_feature = concat_feature.reshape(clip_num,-1,concat_feature.size(-2),concat_feature.size(-1))
        
        # concat_feature = concat_feature.permute(1,0,2,3)
        # concat_feature = [ bz , clip_num ,48 ,1024]
        # concat_feature_view = concat_feature.view(batch_size,clip_num,-1)

        # concat_feature_view = [bz , clip_num , hidden_features]
        # 这一步转换是为了后面能方便地求clip1~8之间的相似度e
        # visual_output = visual_output.permute(1,0,2,3)
        # ——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
        for _idx in range(batch_size):
            single_similarity_m = torch.zeros(clip_num,clip_num)

            for j in range(clip_num):
                for inner_idx in range(j,clip_num):
                    single_similarity_m[j][inner_idx] = torch.cosine_similarity(concat_feature[_idx][j],concat_feature[_idx][inner_idx],dim=0)
            single_similarity_m = single_similarity_m + single_similarity_m.t()
            for j in range(clip_num):
                single_similarity_m[j][j]=0.0
            single_similarity_sft = F.softmax(single_similarity_m,1)
            similarity_matrix[_idx] = single_similarity_sft
        
        
        # softmax normalization
        # similarity = (bz , bz)
        return similarity_matrix
    def _get_decoder_score(self, sequence_output, visual_output, input_ids, attention_mask, video_mask, input_caption_ids, decoder_mask, shaped=False):
        # visual_output
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

            input_caption_ids = input_caption_ids.view(-1, input_caption_ids.shape[-1])
            decoder_mask = decoder_mask.view(-1, decoder_mask.shape[-1])

        res_tuples = ()
        clip_num = visual_output.size(0) // sequence_output.size(0)
        visual_output = visual_output.to(sequence_output.device)
        cross_output, pooled_output, concat_mask = self._get_cross_output(sequence_output, visual_output, attention_mask, video_mask)
        
        input_caption_ids = input_caption_ids.repeat(clip_num,1)
        decoder_mask = decoder_mask.repeat(clip_num,1)
        decoder_scores = self.decoder(input_caption_ids, encoder_outs=cross_output, answer_mask=decoder_mask, encoder_mask=concat_mask)
        
        return decoder_scores,res_tuples
    def _get_decoder_score_siamese(self, sequence_output, visual_output, input_ids, attention_mask, video_mask, input_caption_ids, decoder_mask, shaped=False):

        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

            input_caption_ids = input_caption_ids.view(-1, input_caption_ids.shape[-1])
            decoder_mask = decoder_mask.view(-1, decoder_mask.shape[-1])

        res_tuples = ()
        cross_output, pooled_output, concat_mask = self._get_cross_output(sequence_output, visual_output, attention_mask, video_mask)
        
        decoder_scores = self.decoder(input_caption_ids, encoder_outs=cross_output, answer_mask=decoder_mask, encoder_mask=concat_mask)
        
        siamese_similarity = self.get_crossPair_similarity_logits(sequence_output,visual_output)
        siamese_similarity = siamese_similarity.to(decoder_scores.device)
        decoder_scores_sia = torch.mm(siamese_similarity,decoder_scores.view(siamese_similarity.size(0),-1))
        decoder_scores_sia = decoder_scores_sia.view(decoder_scores.size(0),decoder_scores.size(1),decoder_scores.size(2))
        
        return decoder_scores, decoder_scores_sia,res_tuples

    def decoder_caption(self, sequence_output, visual_output, input_ids, attention_mask, video_mask, input_caption_ids, decoder_mask,
                        shaped=False, get_logits=False):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

            input_caption_ids = input_caption_ids.view(-1, input_caption_ids.shape[-1])
            decoder_mask = decoder_mask.view(-1, decoder_mask.shape[-1])

        decoder_scores, _ = self._get_decoder_score(sequence_output, visual_output,
                                                    input_ids, attention_mask, video_mask,
                                                    input_caption_ids, decoder_mask, shaped=True)

        if get_logits:
            return decoder_scores

        _, decoder_scores_result = torch.max(decoder_scores, -1)

        return decoder_scores_result
    