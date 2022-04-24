from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
from torch.utils.data import Dataset
import numpy as np
import pickle
import pandas as pd
from collections import defaultdict
import json
import h5py
import random
# from data_utils_ACL import *

import argparse, sys, os, io, base64, math

from datetime import datetime
from tqdm import tqdm

import numpy as np
import torch as T
import torchvision as TV
import torch.distributed as DIST

import cv2
from PIL import Image

import transformers
class MSRVTT_Caption_DataLoader_Swin(Dataset):
    """MSRVTT train dataset loader."""
    def __init__(
            self,
            csv_path,
            json_path,
            features_path,
            tokenizer,
            max_words=30,
            feature_framerate=1.0,
            max_frames=100,
            split_type=""
    ):
        self.csv = pd.read_csv(csv_path)
        self.data = json.load(open(json_path, 'r'))
        self.feature_dict = pickle.load(open(features_path, 'rb'))
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer
        self.frame = 4
        self.feature_size = self.feature_dict[self.csv['video_id'].values[0]].shape[-1]
        self.img = pickle.load(open('/home/gujiayang/data/dataset/videocaption/msrvtt_swin/msrvtt_all_6in1.pkl', 'rb'))

        assert split_type in ["train", "val", "test"]
        # Train: video0 : video6512 (6513)
        # Val: video6513 : video7009 (497)
        # Test: video7010 : video9999 (2990)
        video_ids = [self.data['videos'][idx]['video_id'] for idx in range(len(self.data['videos']))]
        split_dict = {"train": video_ids[:6513], "val": video_ids[6513:6513 + 497], "test": video_ids[6513 + 497:]}
        choiced_video_ids = split_dict[split_type]

        self.sample_len = 0
        self.sentences_dict = {}
        self.video_sentences_dict = defaultdict(list)
        if split_type == "train":  # expand all sentence to train
            for itm in self.data['sentences']:
                if itm['video_id'] in choiced_video_ids:
                    self.sentences_dict[len(self.sentences_dict)] = (itm['video_id'], itm['caption'])
                    self.video_sentences_dict[itm['video_id']].append(itm['caption']) # dict存储video序号和video对应的caption
        elif split_type == "val" or split_type == "test":
            for itm in self.data['sentences']:
                if itm['video_id'] in choiced_video_ids:
                    self.video_sentences_dict[itm['video_id']].append(itm['caption'])
            for vid in choiced_video_ids:
                self.sentences_dict[len(self.sentences_dict)] = (vid, self.video_sentences_dict[vid][0]) # sentence dict将上一步的video_text pair作为一个pair，存储进一个array
        else:
            raise NotImplementedError

        self.sample_len = len(self.sentences_dict)

    def __len__(self):
        return self.sample_len

    def _get_text(self, video_id, caption=None):
        k = 1
        choice_video_ids = [video_id]
        pairs_text = np.zeros((k, self.max_words), dtype=np.long)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.long)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.long)
        pairs_masked_text = np.zeros((k, self.max_words), dtype=np.long)
        pairs_token_labels = np.zeros((k, self.max_words), dtype=np.long)

        pairs_input_caption_ids = np.zeros((k, self.max_words), dtype=np.long)
        pairs_output_caption_ids = np.zeros((k, self.max_words), dtype=np.long)
        pairs_decoder_mask = np.zeros((k, self.max_words), dtype=np.long)

        for i, video_id in enumerate(choice_video_ids):
            words = []
            words = ["[CLS]"] + words
            total_length_with_CLS = self.max_words - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + ["[SEP]"]

            # Mask Language Model <-----
            token_labels = []
            masked_tokens = words.copy()
            for token_id, token in enumerate(masked_tokens):
                if token_id == 0 or token_id == len(masked_tokens) - 1:
                    token_labels.append(-1)
                    continue
                prob = random.random()
                # mask token with 15% probability
                if prob < 0.15:
                    prob /= 0.15
                    # 80% randomly change token to mask token
                    if prob < 0.8:
                        masked_tokens[token_id] = "[MASK]"
                    # 10% randomly change token to random token
                    elif prob < 0.9:
                        masked_tokens[token_id] = random.choice(list(self.tokenizer.vocab.items()))[0]
                    # -> rest 10% randomly keep current token
                    # append current token to output (we will predict these later)
                    try:
                        token_labels.append(self.tokenizer.vocab[token])
                    except KeyError:
                        # For unknown words (should not occur with BPE vocab)
                        token_labels.append(self.tokenizer.vocab["[UNK]"])
                        # print("Cannot find token '{}' in vocab. Using [UNK] insetad".format(token))
                else:
                    # no masking token (will be ignored by loss function later)
                    token_labels.append(-1)
            # -----> Mask Language Model

            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
            masked_token_ids = self.tokenizer.convert_tokens_to_ids(masked_tokens)
            while len(input_ids) < self.max_words:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
                masked_token_ids.append(0)
                token_labels.append(-1)
            assert len(input_ids) == self.max_words
            assert len(input_mask) == self.max_words
            assert len(segment_ids) == self.max_words
            assert len(masked_token_ids) == self.max_words
            assert len(token_labels) == self.max_words

            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = np.array(input_mask)
            pairs_segment[i] = np.array(segment_ids)
            pairs_masked_text[i] = np.array(masked_token_ids)
            pairs_token_labels[i] = np.array(token_labels)

            # For generate captions
            if caption is not None:
                caption_words = self.tokenizer.tokenize(caption)
            else:
                caption_words = self._get_single_text(video_id)
            if len(caption_words) > total_length_with_CLS:
                caption_words = caption_words[:total_length_with_CLS]
            input_caption_words = ["[CLS]"] + caption_words
            output_caption_words = caption_words + ["[SEP]"]

            # For generate captions
            input_caption_ids = self.tokenizer.convert_tokens_to_ids(input_caption_words)
            output_caption_ids = self.tokenizer.convert_tokens_to_ids(output_caption_words)
            decoder_mask = [1] * len(input_caption_ids)
            while len(input_caption_ids) < self.max_words:
                input_caption_ids.append(0)
                output_caption_ids.append(0)
                decoder_mask.append(0)
            assert len(input_caption_ids) == self.max_words
            assert len(output_caption_ids) == self.max_words
            assert len(decoder_mask) == self.max_words

            pairs_input_caption_ids[i] = np.array(input_caption_ids)
            pairs_output_caption_ids[i] = np.array(output_caption_ids)
            pairs_decoder_mask[i] = np.array(decoder_mask)
        # pairs_input_captions_ids = CLS+ word_token_id;output同理；
        return pairs_text, pairs_mask, pairs_segment, pairs_masked_text, pairs_token_labels, \
               pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids, choice_video_ids

    def _get_single_text(self, video_id):
        rind = random.randint(0, len(self.sentences[video_id]) - 1)
        caption = self.sentences[video_id][rind]
        words = self.tokenizer.tokenize(caption)
        return words
    
    def video_pad(sequence, max_length):
        sequence = np.array(sequence)
        sequence_shape = np.array(sequence).shape
        current_length = sequence_shape[0]
        pad = np.zeros((max_length, sequence_shape[1]),dtype=np.float32)
        num_padding = max_length - current_length
        if num_padding <= 0:
            pad = sequence[:max_length]
        else:
            pad[:current_length] = sequence
        return pad
    def get_video_feature_ACL(self, video_id): # key : gif_name
        video_id = 'video' + str(video_id)
        i3d_avg_file = os.path.join("/home/lixiangpeng/data/dataset/videocaption/msrvtt_i3d/msrvtt_i3d_avgpool_perclip.hdf5")
        i3d_avg_feat = h5py.File(i3d_avg_file, 'r')
        
        try:
            i3d_avg_feat = np.array(self.i3d_avg_feat[video_id])  # T, d
            # res_roi_feat = np.array(self.res_roi_feat['image_features'][video_id])  # T, 5, d
            # roi_bbox_feat = np.array(self.res_roi_feat['spatial_features'][video_id])  # T, 5, 6
            # i3d_roi_feat = np.array(self.i3d_roi_feat[video_id])  # T, 5, d
        except KeyError: # no img
            print('no image', video_id)
            i3d_avg_feat = np.zeros((1, 2048))
            res_roi_feat = np.zeros((1, self.obj_max_num, 2048))
            roi_bbox_feat = np.zeros((1, self.obj_max_num, 6))
            i3d_roi_feat = np.zeros((1, self.obj_max_num, 2048))
        i3d_avg_pad = self.video_pad(i3d_avg_feat, self.v_max_length).astype(np.float32)

        return  i3d_avg_pad
    
    def _get_video(self, choice_video_ids):
        video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.long)
        max_video_length = [0] * len(choice_video_ids)

        video = np.zeros((len(choice_video_ids), self.max_frames, self.feature_size), dtype=np.float)
        siamese_video = np.zeros((5,len(choice_video_ids), self.max_frames, self.feature_size), dtype=np.float)
        for i, video_id in enumerate(choice_video_ids):
            video_slice = self.feature_dict[video_id]

            if self.max_frames < video_slice.shape[0]:
                video_slice = video_slice[:self.max_frames]

            slice_shape = video_slice.shape
            max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_shape[0] else slice_shape[0]
            if len(video_slice) < 1:
                print("video_id: {}".format(video_id))
            else:
                video[i][:slice_shape[0]] = video_slice

        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length

        # Mask Frame Model <-----
        video_labels_index = [[] for _ in range(len(choice_video_ids))]
        masked_video = video.copy()
        for i, video_pair_ in enumerate(masked_video):
            for j, _ in enumerate(video_pair_):
                if j < max_video_length[i]:
                    prob = random.random()
                    # mask token with 15% probability
                    if prob < 0.15:
                        masked_video[i][j] = [0.] * video.shape[-1]
                        video_labels_index[i].append(j)
                    else:
                        video_labels_index[i].append(-1)
                else:
                    video_labels_index[i].append(-1)
        video_labels_index = np.array(video_labels_index, dtype=np.long)
        # -----> Mask Frame Model

        return video, video_mask, masked_video, video_labels_index
    def str2img(self, b,video_id , caption):
        raw_img = Image.open(io.BytesIO(base64.b64decode(b))).convert('RGB')
        w, h = raw_img.size
        raw_img.save('./4.jpg')
        # img = TV.transforms.Compose([TV.transforms.Resize([224, 224]), 
        #                              TV.transforms.ToTensor()])(raw_img)
        # img = TV.transforms.Compose([TV.transforms.CenterCrop([224, 224]), 
        #                              TV.transforms.ToTensor()])(raw_img)
        
        img = TV.transforms.Compose([TV.transforms.Pad([0, (w-h)//2] if w>h else [(h-w)//2, 0]), 
                                     TV.transforms.Resize([224, 224]), 
                                     TV.transforms.ToTensor()])(raw_img)
        return img
    def __getitem__(self, idx):
        video_id, caption = self.sentences_dict[idx]

        pairs_text, pairs_mask, pairs_segment, \
        pairs_masked_text, pairs_token_labels, \
        pairs_input_caption_ids, pairs_decoder_mask, \
        pairs_output_caption_ids, choice_video_ids = self._get_text(video_id, caption)
        # video = [1,48,1024]
        video, video_mask, masked_video, video_labels_index = self._get_video(choice_video_ids)
        

        
        img = []
        for b in self.img[video_id]:
            img.append(self.str2img(b,video_id,caption).unsqueeze(0))
        
        img = T.cat(img, dim=0)
        # random sampling 48 frames in all frames
        # 先进行一个补全
        if img.size(0)<48:
            for _idx in range(50-img.size(0)):
                img = T.cat((img,img[-1].unsqueeze(0)),dim=0)
        indices = T.Tensor(np.random.choice(img.size(0), 48, replace=False)).long()
        indices , _ = indices.sort()
        img_view = img.reshape(img.size(0),-1)
        img = T.gather(img_view,dim=0,index=indices.unsqueeze(1).expand(48,img_view.size(-1))).reshape(48,img.size(1),img.size(2),img.size(3))
        
        # siamese_video = [ bz, Siamese_clips , frame_max_length , hidden_features ]
        return pairs_text, pairs_mask, pairs_segment, img, video_mask, \
               pairs_masked_text, pairs_token_labels, masked_video, video_labels_index, \
               pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids


class MSRVTT_Caption_DataLoader(Dataset):
    """MSRVTT train dataset loader."""
    def __init__(
            self,
            csv_path,
            json_path,
            features_path,
            tokenizer,
            max_words=30,
            feature_framerate=1.0,
            max_frames=100,
            split_type=""
    ):
        self.csv = pd.read_csv(csv_path)
        self.data = json.load(open(json_path, 'r'))
        self.feature_dict = pickle.load(open(features_path, 'rb'))
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer

        self.feature_size = self.feature_dict[self.csv['video_id'].values[0]].shape[-1]
        self.img = pickle.load(open('/home/gujiayang/data/dataset/videocaption/msrvtt_swin/msrvtt_all.pkl', 'rb'))

        assert split_type in ["train", "val", "test"]
        # Train: video0 : video6512 (6513)
        # Val: video6513 : video7009 (497)
        # Test: video7010 : video9999 (2990)
        video_ids = [self.data['videos'][idx]['video_id'] for idx in range(len(self.data['videos']))]
        split_dict = {"train": video_ids[:6513], "val": video_ids[6513:6513 + 497], "test": video_ids[6513 + 497:]}
        choiced_video_ids = split_dict[split_type]

        self.sample_len = 0
        self.sentences_dict = {}
        self.video_sentences_dict = defaultdict(list)
        if split_type == "train":  # expand all sentence to train
            for itm in self.data['sentences']:
                if itm['video_id'] in choiced_video_ids:
                    self.sentences_dict[len(self.sentences_dict)] = (itm['video_id'], itm['caption'])
                    self.video_sentences_dict[itm['video_id']].append(itm['caption']) # dict存储video序号和video对应的caption
        elif split_type == "val" or split_type == "test":
            for itm in self.data['sentences']:
                if itm['video_id'] in choiced_video_ids:
                    self.video_sentences_dict[itm['video_id']].append(itm['caption'])
            for vid in choiced_video_ids:
                self.sentences_dict[len(self.sentences_dict)] = (vid, self.video_sentences_dict[vid][0]) # sentence dict将上一步的video_text pair作为一个pair，存储进一个array
        else:
            raise NotImplementedError

        self.sample_len = len(self.sentences_dict)

    def __len__(self):
        return self.sample_len

    def _get_text(self, video_id, caption=None):
        k = 1
        choice_video_ids = [video_id]
        pairs_text = np.zeros((k, self.max_words), dtype=np.long)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.long)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.long)
        pairs_masked_text = np.zeros((k, self.max_words), dtype=np.long)
        pairs_token_labels = np.zeros((k, self.max_words), dtype=np.long)

        pairs_input_caption_ids = np.zeros((k, self.max_words), dtype=np.long)
        pairs_output_caption_ids = np.zeros((k, self.max_words), dtype=np.long)
        pairs_decoder_mask = np.zeros((k, self.max_words), dtype=np.long)

        for i, video_id in enumerate(choice_video_ids):
            words = []
            words = ["[CLS]"] + words
            total_length_with_CLS = self.max_words - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + ["[SEP]"]

            # Mask Language Model <-----
            token_labels = []
            masked_tokens = words.copy()
            for token_id, token in enumerate(masked_tokens):
                if token_id == 0 or token_id == len(masked_tokens) - 1:
                    token_labels.append(-1)
                    continue
                prob = random.random()
                # mask token with 15% probability
                if prob < 0.15:
                    prob /= 0.15
                    # 80% randomly change token to mask token
                    if prob < 0.8:
                        masked_tokens[token_id] = "[MASK]"
                    # 10% randomly change token to random token
                    elif prob < 0.9:
                        masked_tokens[token_id] = random.choice(list(self.tokenizer.vocab.items()))[0]
                    # -> rest 10% randomly keep current token
                    # append current token to output (we will predict these later)
                    try:
                        token_labels.append(self.tokenizer.vocab[token])
                    except KeyError:
                        # For unknown words (should not occur with BPE vocab)
                        token_labels.append(self.tokenizer.vocab["[UNK]"])
                        # print("Cannot find token '{}' in vocab. Using [UNK] insetad".format(token))
                else:
                    # no masking token (will be ignored by loss function later)
                    token_labels.append(-1)
            # -----> Mask Language Model

            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
            masked_token_ids = self.tokenizer.convert_tokens_to_ids(masked_tokens)
            while len(input_ids) < self.max_words:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
                masked_token_ids.append(0)
                token_labels.append(-1)
            assert len(input_ids) == self.max_words
            assert len(input_mask) == self.max_words
            assert len(segment_ids) == self.max_words
            assert len(masked_token_ids) == self.max_words
            assert len(token_labels) == self.max_words

            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = np.array(input_mask)
            pairs_segment[i] = np.array(segment_ids)
            pairs_masked_text[i] = np.array(masked_token_ids)
            pairs_token_labels[i] = np.array(token_labels)

            # For generate captions
            if caption is not None:
                caption_words = self.tokenizer.tokenize(caption)
            else:
                caption_words = self._get_single_text(video_id)
            if len(caption_words) > total_length_with_CLS:
                caption_words = caption_words[:total_length_with_CLS]
            input_caption_words = ["[CLS]"] + caption_words
            output_caption_words = caption_words + ["[SEP]"]

            # For generate captions
            input_caption_ids = self.tokenizer.convert_tokens_to_ids(input_caption_words)
            output_caption_ids = self.tokenizer.convert_tokens_to_ids(output_caption_words)
            decoder_mask = [1] * len(input_caption_ids)
            while len(input_caption_ids) < self.max_words:
                input_caption_ids.append(0)
                output_caption_ids.append(0)
                decoder_mask.append(0)
            assert len(input_caption_ids) == self.max_words
            assert len(output_caption_ids) == self.max_words
            assert len(decoder_mask) == self.max_words

            pairs_input_caption_ids[i] = np.array(input_caption_ids)
            pairs_output_caption_ids[i] = np.array(output_caption_ids)
            pairs_decoder_mask[i] = np.array(decoder_mask)
        # pairs_input_captions_ids = CLS+ word_token_id;output同理；
        return pairs_text, pairs_mask, pairs_segment, pairs_masked_text, pairs_token_labels, \
               pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids, choice_video_ids

    def _get_single_text(self, video_id):
        rind = random.randint(0, len(self.sentences[video_id]) - 1)
        caption = self.sentences[video_id][rind]
        words = self.tokenizer.tokenize(caption)
        return words
    
    def video_pad(sequence, max_length):
        sequence = np.array(sequence)
        sequence_shape = np.array(sequence).shape
        current_length = sequence_shape[0]
        pad = np.zeros((max_length, sequence_shape[1]),dtype=np.float32)
        num_padding = max_length - current_length
        if num_padding <= 0:
            pad = sequence[:max_length]
        else:
            pad[:current_length] = sequence
        return pad
    def get_video_feature_ACL(self, video_id): # key : gif_name
        video_id = 'video' + str(video_id)
        i3d_avg_file = os.path.join("/home/lixiangpeng/data/dataset/videocaption/msrvtt_i3d/msrvtt_i3d_avgpool_perclip.hdf5")
        i3d_avg_feat = h5py.File(i3d_avg_file, 'r')
        
        try:
            i3d_avg_feat = np.array(self.i3d_avg_feat[video_id])  # T, d
            # res_roi_feat = np.array(self.res_roi_feat['image_features'][video_id])  # T, 5, d
            # roi_bbox_feat = np.array(self.res_roi_feat['spatial_features'][video_id])  # T, 5, 6
            # i3d_roi_feat = np.array(self.i3d_roi_feat[video_id])  # T, 5, d
        except KeyError: # no img
            print('no image', video_id)
            i3d_avg_feat = np.zeros((1, 2048))
            res_roi_feat = np.zeros((1, self.obj_max_num, 2048))
            roi_bbox_feat = np.zeros((1, self.obj_max_num, 6))
            i3d_roi_feat = np.zeros((1, self.obj_max_num, 2048))
        i3d_avg_pad = self.video_pad(i3d_avg_feat, self.v_max_length).astype(np.float32)

        return  i3d_avg_pad
    
    def _get_video(self, choice_video_ids):
        video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.long)
        max_video_length = [0] * len(choice_video_ids)

        video = np.zeros((len(choice_video_ids), self.max_frames, self.feature_size), dtype=np.float)
        siamese_video = np.zeros((5,len(choice_video_ids), self.max_frames, self.feature_size), dtype=np.float)
        for i, video_id in enumerate(choice_video_ids):
            video_slice = self.feature_dict[video_id]

            if self.max_frames < video_slice.shape[0]:
                video_slice = video_slice[:self.max_frames]

            slice_shape = video_slice.shape
            max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_shape[0] else slice_shape[0]
            if len(video_slice) < 1:
                print("video_id: {}".format(video_id))
            else:
                video[i][:slice_shape[0]] = video_slice

        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length

        # Mask Frame Model <-----
        video_labels_index = [[] for _ in range(len(choice_video_ids))]
        masked_video = video.copy()
        for i, video_pair_ in enumerate(masked_video):
            for j, _ in enumerate(video_pair_):
                if j < max_video_length[i]:
                    prob = random.random()
                    # mask token with 15% probability
                    if prob < 0.15:
                        masked_video[i][j] = [0.] * video.shape[-1]
                        video_labels_index[i].append(j)
                    else:
                        video_labels_index[i].append(-1)
                else:
                    video_labels_index[i].append(-1)
        video_labels_index = np.array(video_labels_index, dtype=np.long)
        # -----> Mask Frame Model

        return video, video_mask, masked_video, video_labels_index

    def __getitem__(self, idx):
        video_id, caption = self.sentences_dict[idx]

        pairs_text, pairs_mask, pairs_segment, \
        pairs_masked_text, pairs_token_labels, \
        pairs_input_caption_ids, pairs_decoder_mask, \
        pairs_output_caption_ids, choice_video_ids = self._get_text(video_id, caption)

        video, video_mask, masked_video, video_labels_index = self._get_video(choice_video_ids)
        # siamese_video = [ bz, Siamese_clips , frame_max_length , hidden_features ]
        return pairs_text, pairs_mask, pairs_segment, video, video_mask, \
               pairs_masked_text, pairs_token_labels, masked_video, video_labels_index, \
               pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids
