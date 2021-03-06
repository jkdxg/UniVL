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
import torchvision as TV
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss

from modules.until_module import PreTrainedModel, LayerNorm, CrossEn, MILNCELoss, MaxMarginRankingLoss
from modules.module_bert import BertModel, BertConfig, BertOnlyMLMHead 
from transformers import BertTokenizer
from modules.module_visual import VisualModel, VisualConfig, VisualOnlyMLMHead
from modules.module_cross import CrossModel, CrossConfig
from modules.module_decoder import DecoderModel, DecoderConfig
from torch.autograd import Variable
from modules.vid_swin import SwinTransformer3D
logger = logging.getLogger(__name__)

import transformers


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
        
        self.clip_num = 3
        self.frame_num = 6
        self.emb_cls = torch.nn.Parameter(0.02*torch.randn(1, 1, 1, 768))
        self.emb_pos = torch.nn.Parameter(0.02*torch.randn(1, 1, 1+14**2, 768))
        self.emb_len = torch.nn.Parameter(0.02*torch.randn(1, 6, 1, 768))
        
        bert = transformers.BertForMaskedLM.from_pretrained('bert-base-uncased')
        self.mask_ext, self.trsfr = bert.get_extended_attention_mask, bert.bert.encoder

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
        self.swin = SwinTransformer3D()
        self.swin.load_state_dict(torch.load('/home/gujiayang/workspace/videocaption/UniVL/_snapshot/ckpt_video-swin.pt', map_location='cpu'))
        self.swinNorm = torch.nn.LayerNorm(768)
        
        self.fc1 = torch.nn.Linear(self.clip_num,1)
        self.fc2 = torch.nn.Linear(self.clip_num,1)
        self.swin_trigger = True
        self.siamese_trigger = True
        visual_word_embeddings_weight = self.visual.embeddings.word_embeddings.weight
        # <=== End of Video Encoder
        
        #BERT-Decoder
        # self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # config = transformers.BertConfig.from_pretrained("bert-base-uncased", bos_token_id=101, pad_token_id=0, eos_token_id=102)
        # config.is_decoder = True
        # config.add_cross_attention = True
        # config.hidden_dropout_prob = 0.3
        # config.attention_probs_dropout_prob = 0.3
        # self.decoder = transformers.BertLMHeadModel.from_pretrained('bert-base-uncased', config=config)
        
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
        self.weighted_m1 = torch.nn.Parameter(torch.ones(task_config.batch_size// task_config.n_gpu,1,self.clip_num),requires_grad = True)
        self.weighted_m2 = torch.nn.Parameter(torch.ones(task_config.batch_size// task_config.n_gpu,1,self.clip_num),requires_grad = True)
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
                input_caption_ids=None, decoder_mask=None, output_caption_ids=None,siamese_trigger = False,swin_trigger = True,tokenizer = None,output_trigger=False):
        starttime = time.perf_counter()
        self.swin_trigger = swin_trigger
        input_ids = input_ids.view(-1, input_ids.shape[-1])
        token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
        attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
        video_mask = video_mask.view(-1, video_mask.shape[-1])
        self.siamese_trigger = siamese_trigger
        # for _idx in range(siamese_clip_num):
        #     siamese_video[_idx] = self.normalize_video(siamese_video[_idx])
        
        # print(f'nomalized:{time.perf_counter() - starttime:.8f}s')
        
        starttime = time.perf_counter()
        if input_caption_ids is not None:
            input_caption_ids = input_caption_ids.view(-1, input_caption_ids.shape[-1])
            decoder_mask = decoder_mask.view(-1, decoder_mask.shape[-1])
        # visual_output = [siamese_num , bz , frm, hidden_layer]
        if swin_trigger == False:
            video = self.normalize_video(video)
            sequence_output, visual_output = self.get_sequence_visual_output(input_ids, token_type_ids, attention_mask,
                                                                         siamese_video,video, video_mask, siamese_trigger,shaped=True)
        else:
            video_mask = video_mask.repeat(1,50)
            sequence_output, visual_output = self.get_sequence_visual_output_VT(input_ids, token_type_ids, attention_mask,
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
                            video_mask = torch.ones(video_mask.size(0),self.frame_num*50).to(video_mask.device)
                            decoder_scores,res_tuples = self._get_decoder_score(sequence_output, visual_output,
                                                                             input_ids, attention_mask, video_mask,
                                                                             input_caption_ids, decoder_mask, shaped=True)
                        else:
                            video_mask = torch.ones(video_mask.size(0),self.frame_num*50).to(video_mask.device)
                            decoder_scores_siamese,res_tuples = self._get_decoder_score(sequence_output, visual_output,
                                                                             input_ids, attention_mask, video_mask,
                                                                             input_caption_ids, decoder_mask, shaped=True)
                            # decoder_scores_siamese = [clip_num , bz , frm_max_length , score_features]
                            decoder_scores_siamese = decoder_scores_siamese.reshape(self.clip_num,decoder_scores_siamese.size(0) // self.clip_num , decoder_scores_siamese.size(-2),decoder_scores_siamese.size(-1))
                            decoder_scores = decoder_scores_siamese[0]
                        
                           
                            
                    else:
                        raise NotImplementedError
                    # visual_output = [clipNum , bz , 48 , 1024]
                    
                    output_caption_ids = output_caption_ids.view(-1, output_caption_ids.shape[-1])
                    decoder_loss = self.decoder_loss_fct(decoder_scores.view(-1, self.bert_config.vocab_size), output_caption_ids.view(-1))
                    # tokenizer.convert_ids_to_tokens([7002,7010])
                    if output_trigger ==True:
                        _,ans_idx = torch.max(decoder_scores[-1],1)
                        ans_idx = ans_idx.cpu().numpy().tolist()
                        gth_idx = output_caption_ids[0].cpu().numpy().tolist()
                        gth_caption_list = tokenizer.convert_ids_to_tokens(gth_idx)
                        prd_caption_list = tokenizer.convert_ids_to_tokens(ans_idx)
                        gth_caption=""
                        prd_caption=""
                        temp_char = ''
                        idx = 0
                        while (temp_char != '[PAD]' and idx<=40):
                            temp_char=gth_caption_list[idx]
                            idx+=1
                            gth_caption = gth_caption+' '+temp_char 
                            
                        temp_char = ''
                        idx = 0
                        while (temp_char != '[PAD]'and idx<=40):
                            temp_char=prd_caption_list[idx]
                            idx+=1
                            prd_caption = prd_caption+' '+temp_char 
                            
                        print("GTH : %s"%gth_caption)
                        print("PRD : %s"%prd_caption)
                        print("-------------------")
                    
                    siamese_loss = 0.0
                    loss = decoder_loss
                    if siamese_trigger ==True:
                        siamese_loss = self.siamese_reasoning(sequence_output,visual_output,decoder_scores_siamese)
                        loss =loss +siamese_loss
                    

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

    def siamese_reasoning(self,sequence_output,visual_output,decoder_scores_siamese):
        siamese_similarity_m = self.get_crossPair_similarity_logits_concat(sequence_output , visual_output,self.frame_num).to(sequence_output.device)
        decoder_scores_siamese_T = decoder_scores_siamese.permute(1,0,2,3)
                        
        dec_soft_label = torch.bmm(siamese_similarity_m,decoder_scores_siamese_T.view(decoder_scores_siamese_T.size(0),decoder_scores_siamese_T.size(1),-1))
        dec_soft_label = torch.reshape(dec_soft_label,decoder_scores_siamese_T.size())

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
                        
        # ???learning from inside???????????????anchor???loss
        siamese_loss = self.decoder_loss_fct(final_score_pseudo.view(-1,self.bert_config.vocab_size),GTH_index.view(-1))
        return siamese_loss
    
    def go_cross(self, feat_img, mask_img, feat_txt, mask_txt,clip_num):
        if (self.siamese_trigger):
            mask_img , mask_txt = mask_img.repeat(clip_num,1) , mask_txt.repeat(clip_num,1)
        feat, mask = torch.cat([feat_img, feat_txt], dim=1), torch.cat([mask_img, mask_txt], dim=1)
        mask = self.mask_ext(mask, mask.shape, mask.device)
        out = self.trsfr(feat, mask, output_attentions=True)
        return out['last_hidden_state'], out['attentions']
   
    
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
        siamese_output = torch.zeros(video.size(0),self.clip_num,self.frame_num,video.size(-1))
        sia_idx = np.zeros((self.clip_num,self.frame_num),int)
        single_ran_idx = np.zeros(self.frame_num,int)
        for j in range(self.frame_num):
            if j ==0:
                single_ran_idx[j] = np.random.randint(0,self.frame_num*j+self.frame_num)
            elif j<=2:
                single_ran_idx[j] = np.random.randint(single_ran_idx[j-1]+1,self.frame_num*j+self.frame_num)
            else:
                single_ran_idx[j] = np.random.randint(single_ran_idx[j-1]+1,max_frames-1-self.frame_num-5+j)
        sia_idx[0]=single_ran_idx
        for i in range(1,self.clip_num):
            sia_idx[i] = i+sia_idx[0]
        sia_idx = torch.from_numpy(sia_idx)
        # For Indexing
        sia_idx = torch.LongTensor(sia_idx).to(video.device)
        for batch_num in range(video.size(0)):
        
            single_batch_video = video[batch_num]
            # gathered_batch_video = [clip_num , 5 , 1024]
            gathered_batch_video = single_batch_video.unsqueeze(1).expand(48,self.frame_num,768).gather(dim = 0 ,index=sia_idx.unsqueeze(2).expand(self.clip_num,self.frame_num,768))
            # gathered_batch_video = [clip_num, 5 + padding=48 , 1024] == [ 8 , 48 , 1024 ] 
            # need to padding second dim to match frm_max_length
            # gathered_batch_video_padding = padding_video(gathered_batch_video,max_frames)

            siamese_output[batch_num] = gathered_batch_video
    
    
        siamese_video = siamese_output.permute(1,0,2,3)
        # output = [8 , bz , 48 , 1024]
        return siamese_video
    def getClips_VST(self,video,max_frames):
        sia_idx = np.zeros((self.clip_num,self.frame_num),int)
        single_ran_idx = np.zeros(self.frame_num,int)
        for j in range(self.frame_num):
            if j ==0:
                single_ran_idx[j] = np.random.randint(0,self.frame_num*j+self.frame_num)
            elif j<=2:
                single_ran_idx[j] = np.random.randint(single_ran_idx[j-1]+1,self.frame_num*j+self.frame_num)
            else:
                single_ran_idx[j] = np.random.randint(single_ran_idx[j-1]+1,max_frames-1-self.frame_num-5+j)
        sia_idx[0]=single_ran_idx
        for i in range(1,self.clip_num):
            sia_idx[i] = i+sia_idx[0]
        sia_idx = torch.from_numpy(sia_idx)
        # For Indexing
        sia_idx = torch.LongTensor(sia_idx).to(video.device)
        # video_view = [bz,48,hidden_state]
        video_view = video.view(video.size(0),video.size(1),-1)
        hidden_dim = video_view.size(-1)
        siamese_output = torch.zeros(video.size(0),self.clip_num,self.frame_num,hidden_dim)
        
        for batch_num in range(video.size(0)):
        
            single_batch_video = video_view[batch_num]
            # gathered_batch_video = [clip_num , 5 , 1024]
            gathered_batch_video = single_batch_video.unsqueeze(1).expand(48,self.frame_num,hidden_dim).gather(dim = 0 ,index=sia_idx.unsqueeze(2).expand(self.clip_num,self.frame_num,hidden_dim))
            # gathered_batch_video = [clip_num, 5 + padding=48 , 1024] == [ 8 , 48 , 1024 ] 
            # need to padding second dim to match frm_max_length
            # gathered_batch_video_padding = padding_video(gathered_batch_video,max_frames)

            siamese_output[batch_num] = gathered_batch_video
    
        # siamese_output = [bz , clip , frm , hidden]     
        siamese_video = siamese_output.permute(1,0,2,3)
        # siamese_video = [clip , bz , 30 , 3 , 224 ,224]  
        siamese_video = siamese_video.view(siamese_video.size(0),siamese_video.size(1),siamese_video.size(2),video.size(-3),video.size(-2),video.size(-1))
        
        
        return siamese_video[0]
    def getSiameseClips_VST(self,video,max_frames):
        # video = [bz , 48 , 3 , 224 , 224]
        
        sia_idx = np.zeros((self.clip_num,self.frame_num),int)
        single_ran_idx = np.zeros(self.frame_num,int)
        for j in range(self.frame_num):
            if j ==0:
                single_ran_idx[j] = np.random.randint(0,self.frame_num*j+self.frame_num)
            elif j<=2:
                single_ran_idx[j] = np.random.randint(single_ran_idx[j-1]+1,self.frame_num*j+self.frame_num)
            else:
                single_ran_idx[j] = np.random.randint(single_ran_idx[j-1]+1,max_frames-1-self.frame_num-5+j)
        sia_idx[0]=single_ran_idx
        for i in range(1,self.clip_num):
            sia_idx[i] = i+sia_idx[0]
        sia_idx = torch.from_numpy(sia_idx)
        # For Indexing
        sia_idx = torch.LongTensor(sia_idx).to(video.device)
        # video_view = [bz,48,hidden_state]
        video_view = video.view(video.size(0),video.size(1),-1)
        hidden_dim = video_view.size(-1)
        siamese_output = torch.zeros(video.size(0),self.clip_num,self.frame_num,hidden_dim)
        
        for batch_num in range(video.size(0)):
        
            single_batch_video = video_view[batch_num]
            # gathered_batch_video = [clip_num , 5 , 1024]
            gathered_batch_video = single_batch_video.unsqueeze(1).expand(48,self.frame_num,hidden_dim).gather(dim = 0 ,index=sia_idx.unsqueeze(2).expand(self.clip_num,self.frame_num,hidden_dim))
            # gathered_batch_video = [clip_num, 5 + padding=48 , 1024] == [ 8 , 48 , 1024 ] 
            # need to padding second dim to match frm_max_length
            # gathered_batch_video_padding = padding_video(gathered_batch_video,max_frames)

            siamese_output[batch_num] = gathered_batch_video
    
        # siamese_output = [bz , clip , frm , hidden]     
        siamese_video = siamese_output.permute(1,0,2,3)
        # siamese_video = [clip , bz , 30 , 3 , 224 ,224]  
        siamese_video = siamese_video.view(siamese_video.size(0),siamese_video.size(1),siamese_video.size(2),video.size(-3),video.size(-2),video.size(-1))
        
        
        return siamese_video
    
    def getSiameseClips_VST_eval(self,video,max_frames):
        # video = [bz , 48 , 3 , 224 , 224]
        

        single_ran_idx = np.zeros(self.frame_num,int)
        for j in range(self.frame_num):
            if j ==0:
                single_ran_idx[j] = np.random.randint(0,self.frame_num*j+self.frame_num)
            elif j<=2:
                single_ran_idx[j] = np.random.randint(single_ran_idx[j-1]+1,self.frame_num*j+self.frame_num)
            else:
                single_ran_idx[j] = np.random.randint(single_ran_idx[j-1]+1,max_frames-1-self.frame_num-5+j)
        

        # For Indexing

        single_ran_idx = torch.LongTensor(single_ran_idx).to(video.device)
        # video_view = [bz,48,hidden_state]
        video_view = video.view(video.size(0),video.size(1),-1)
        hidden_dim = video_view.size(-1)
        siamese_output = torch.zeros(video.size(0),self.frame_num,hidden_dim)
        
        for batch_num in range(video.size(0)):
        
            single_batch_video = video_view[batch_num]
            # gathered_batch_video = [clip_num , 5 , 1024]
            gathered_batch_video = single_batch_video.gather(dim = 0 ,index=single_ran_idx.unsqueeze(1).expand(self.frame_num,hidden_dim))
            # gathered_batch_video = [clip_num, 5 + padding=48 , 1024] == [ 8 , 48 , 1024 ] 
            # need to padding second dim to match frm_max_length
            # gathered_batch_video_padding = padding_video(gathered_batch_video,max_frames)

            siamese_output[batch_num] = gathered_batch_video
    
        # siamese_video = [bz , 48 , 1024]     

        # siamese_video = [bz , 30 , 3 , 224 ,224]  
        siamese_video = siamese_output.reshape(video.size(0),siamese_output.size(1),video.size(2),video.size(3),-1)
        
        
        
        return siamese_video
    def get_sequence_visual_output(self, input_ids, token_type_ids, attention_mask, video,raw_video,video_mask, siamese_trigger,shaped=False):

        shaped_video = video.reshape(-1,video.size(-2),video.size(-1))
        # 
        siamese_video = torch.zeros(raw_video.size()).repeat(self.clip_num,1,1)
        clip_num = video.size(0)
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            video = self.normalize_video(video)

        encoded_layers, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=True) # ???????????????12????????????????????????????????????Sequence output
        sequence_output = encoded_layers[-1]
        
        if siamese_trigger == False:
            visual_layers, _ = self.visual(raw_video, video_mask, output_all_encoded_layers=True)
            visual_output = visual_layers[-1]
            return sequence_output,visual_output
        # video = [clip*bz , 5,feature]
        frame_num = video.size(-2)
        # video_mask = torch.ones(video.size(0),self.frame_num).long().to(sequence_output.device)
        video = video.to(sequence_output.device)

        visual_layers, _ = self.visual(raw_video, video_mask, output_all_encoded_layers=True)
        visual_output = visual_layers[-1]
        
        siamese_video = self.getSiameseClips(visual_output,48)
        
        return sequence_output, siamese_video
    
    def get_sequence_visual_output_VT(self, input_ids, token_type_ids, attention_mask, video,raw_video,video_mask, siamese_trigger,shaped=False):
    
        shaped_video = video.reshape(-1,video.size(-2),video.size(-1))
        # 
        # siamese_video = torch.zeros(raw_video.size()).repeat(self.clip_num,1,1)
        encoded_layers, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=True) # ???????????????12????????????????????????????????????Sequence output
        sequence_output = encoded_layers[-1]
        # video_mask = torch.ones(video.size(0),self.frame_num).long().to(sequence_output.device)
        raw_video = raw_video.to(sequence_output.device)

        _B, _T, _C, _H, _W = raw_video.shape
        _h, _w = _H//32, _W//32
        
        raw_video = TV.transforms.Normalize([0.485, 0.456, 0.406], 
                                      [0.229, 0.224, 0.225])(raw_video)
        if siamese_trigger == True:
            selected_video = self.getSiameseClips_VST(raw_video,48)
            selected_video = selected_video.reshape(-1,selected_video.size(2),selected_video.size(3),selected_video.size(4),selected_video.size(5)).to(sequence_output.device)
            f_video = self.swin(selected_video.transpose(1, 2)).transpose(1, 2)
            
            # f_video = f_video.permute(0, 1, 3, 4, 2).reshape(self.clip_num,_B ,-1,_h*_w,768)
            f_video = f_video.permute(0, 1, 3, 4, 2)
            _B,_T,_,_,_ = f_video.shape
            f_video = f_video.reshape(_B,_T,-1,f_video.size(-1))
            f_video = torch.cat([self.emb_cls.expand([_B, _T, -1, -1]), f_video], dim=2)
            f_video += self.emb_pos.expand([_B, _T, -1, -1])[:, :, :1+_h*_w, :]+self.emb_len.expand([_B, -1, 1+_h*_w, -1])[:, :_T, :, :]
            
            f_video = self.swinNorm(f_video)

            # f_video = [bz,frm,49,768]

            # f_video_mean = torch.mean(f_video,-2)
            
            
            
            
            return sequence_output, f_video
        else:
            selected_video = self.getClips_VST(raw_video,48)
            selected_video = selected_video.to(sequence_output.device)
            f_video = self.swin(selected_video.transpose(1, 2)).transpose(1, 2)

            # f_video = f_video.permute(0, 1, 3, 4, 2).reshape(self.clip_num,_B ,-1,_h*_w,768)
            f_video = f_video.permute(0, 1, 3, 4, 2)
            _B,_T,_,_,_ = f_video.shape
            f_video = f_video.reshape(_B,_T,-1,f_video.size(-1))
            f_video = torch.cat([self.emb_cls.expand([_B, _T, -1, -1]), f_video], dim=2)
            f_video += self.emb_pos.expand([_B, _T, -1, -1])[:, :, :1+_h*_w, :]+self.emb_len.expand([_B, -1, 1+_h*_w, -1])[:, :_T, :, :]

            f_video = self.swinNorm(f_video)

            # f_video = [bz,frm,49,768]

            # f_video_mean = torch.mean(f_video,-2)


            # feat_img, mask_img= self.go_feat(img)#torch.Size([6, 250, 768]),torch.Size([6, 250])
            # out, _ = self.go_cross(feat_img, mask_img)#torch.Size([6, 250, 768])
            # out=torch.cat((out[:,0, :].unsqueeze(1),out[:,50, :].unsqueeze(1),out[:,100, :].unsqueeze(1),out[:,150, :].unsqueeze(1),out[:,200, :].unsqueeze(1)),dim=1)
            
            return sequence_output, f_video
        
    
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
    
    def get_sequence_visual_output_eval_swin(self, input_ids, token_type_ids, attention_mask, video, video_mask, siamese_trigger,shaped=False):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            # video = self.normalize_video(video)

        encoded_layers, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=True)
        sequence_output = encoded_layers[-1]

        # visual_layers, _ = self.visual(video, video_mask, output_all_encoded_layers=True)
        # visual_output = visual_layers[-1]
        
        # raw_video = [4,48,3,224,224]
        
        raw_video = video.to(sequence_output.device)

        
        _B, _T, _C, _H, _W = raw_video.shape
        _h, _w = _H//32, _W//32
        
        raw_video = TV.transforms.Normalize([0.485, 0.456, 0.406], 
                                      [0.229, 0.224, 0.225])(raw_video)
        selected_video = self.getSiameseClips_VST_eval(raw_video,48)
        selected_video = selected_video.to(sequence_output.device)
        f_video = self.swin(selected_video.transpose(1, 2)).transpose(1, 2)

        f_video = f_video.permute(0, 1, 3, 4, 2)
        _B,_T,_,_,_ = f_video.shape
        f_video = f_video.reshape(_B,_T,-1,f_video.size(-1))
        f_video = torch.cat([self.emb_cls.expand([_B, _T, -1, -1]), f_video], dim=2)
        f_video += self.emb_pos.expand([_B, _T, -1, -1])[:, :, :1+_h*_w, :]+self.emb_len.expand([_B, -1, 1+_h*_w, -1])[:, :_T, :, :]
        
        f_video = self.swinNorm(f_video)
        return sequence_output, f_video
    
    def _get_cross_output(self, sequence_output, visual_output, attention_mask, video_mask):
        # ?????????visual_output????????????????????????????????????sequence_output??????????????????
        # visual_output_view = visual_output.reshape()
        visual_output = visual_output.reshape(visual_output.size(0),-1,visual_output.size(-1))
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
    def _get_cross_output_swin(self, sequence_output, visual_output, attention_mask, video_mask):
        # ?????????visual_output????????????????????????????????????sequence_output??????????????????
        clip_num = visual_output.size(0) // sequence_output.size(0)
        sequence_output = sequence_output.repeat(clip_num,1,1)
        visual_output = visual_output.reshape(visual_output.size(0),-1,visual_output.size(-1))
        concat_features = torch.cat((sequence_output, visual_output), dim=1)  # concatnate tokens and frames
        
        
        text_type_ = torch.zeros_like(attention_mask)
        video_type_ = torch.ones_like(video_mask)
        concat_type = torch.cat((text_type_, video_type_), dim=1)
        
        concat_type = concat_type.repeat(clip_num,1)
        concat_type = concat_type.long()
        concat_mask = torch.cat((attention_mask, video_mask), dim=1)
        concat_mask = concat_mask.repeat(clip_num,1)
        # -------------------------------- violet cross begin -------------------------------- #
        # out , _ = self.go_cross(visual_output,video_mask,sequence_output,attention_mask,clip_num)
        # out = video + txt
        
        # global_output=torch.cat((out[:,0, :].unsqueeze(1),out[:,50, :].unsqueeze(1),out[:,100, :].unsqueeze(1),out[:,150, :].unsqueeze(1),out[:,200, :].unsqueeze(1),out[:,250, :].unsqueeze(1)),dim=1)
        # c_out = torch.cat((global_output,out[:,300:,:]),dim=1)
        
        # concat_mask = torch.cat((video_mask[:,-(self.frame_num):],attention_mask), dim=1)
        # concat_mask = concat_mask.repeat(clip_num,1)
        # return c_out, concat_mask
        # -------------------------------- violet cross end -------------------------------- #
        
        cross_layers, pooled_output = self.cross(concat_features, concat_type, concat_mask, output_all_encoded_layers=True)
        cross_output = cross_layers[-1]

        return cross_output, concat_mask
        

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
    def get_crossPair_similarity_logits_concat(self, sequence_output, visual_output , max_frm_length = 48):
        # visual_output = [clipNum , bz , 48 , 1024]
        batch_size = sequence_output.size(0)
        sequence_output = sequence_output.repeat(self.clip_num,1,1)
        visual_output = visual_output.reshape(visual_output.size(0),-1,visual_output.size(-1))
        
        max_frm_length = visual_output.size(2)


        similarity_matrix = torch.zeros(batch_size,self.clip_num,self.clip_num)
       
        visual_output = visual_output.to(sequence_output.device)
        
        
        concat_feature = torch.cat((sequence_output,visual_output),dim = 1)  
        concat_feature = concat_feature.reshape(batch_size,self.clip_num,-1)
        
        for _idx in range(batch_size):
            single_similarity_m = torch.zeros(self.clip_num,self.clip_num)

            for j in range(self.clip_num):
                for inner_idx in range(j,self.clip_num):
                    single_similarity_m[j][inner_idx] = torch.cosine_similarity(concat_feature[_idx][j],concat_feature[_idx][inner_idx],dim=0)
            single_similarity_m = single_similarity_m + single_similarity_m.t()
            for j in range(self.clip_num):
                single_similarity_m[j][j]=0.0
            single_similarity_sft = F.softmax(single_similarity_m,1)
            similarity_matrix[_idx] = single_similarity_sft

        return similarity_matrix
    def get_crossPair_similarity_logits(self,cross_input , max_frm_length = 48):
        similarity_matrix = torch.zeros(16,self.clip_num,self.clip_num)
        # ????????????Encoder??????????????????
        # cross_input = [bz*clip , 48+frame , hidden_feature]
        # cross_input_view = cross_input.reshape(self.clip_num,cross_input.size(0) // self.clip_num , -1)
        # cross_input_view = cross_input_view.permute(1,0,2)
        cross_input_view = cross_input.reshape(cross_input.size(0) // self.clip_num , self.clip_num, -1)
        
        # cross_input_view = [16,5,30522]
        
        for _idx in range(cross_input.size(0) // self.clip_num):
            single_similarity_m = torch.zeros(self.clip_num,self.clip_num)

            for j in range(self.clip_num):
                for inner_idx in range(j,self.clip_num):
                    single_similarity_m[j][inner_idx] = torch.cosine_similarity(cross_input_view[_idx][j],cross_input_view[_idx][inner_idx],dim=0)
            single_similarity_m = single_similarity_m + single_similarity_m.t()
            for j in range(self.clip_num):
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
        if self.swin_trigger == True: 
            # cross_output, concat_mask = self._get_cross_output_swin(sequence_output, visual_output, attention_mask, video_mask)
            cross_output, pooled_output, concat_mask = self._get_cross_output(sequence_output, visual_output, attention_mask, video_mask)
        else:
            cross_output, pooled_output, concat_mask = self._get_cross_output(sequence_output, visual_output, attention_mask, video_mask)
        
        input_caption_ids = input_caption_ids.repeat(clip_num,1)
        decoder_mask = decoder_mask.repeat(clip_num,1)
        decoder_scores = self.decoder(input_caption_ids, encoder_outs=cross_output, answer_mask=decoder_mask, encoder_mask=concat_mask)
        
        return decoder_scores,res_tuples
    

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
    