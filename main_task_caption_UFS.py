from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from locale import normalize
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

# os.environ["CUDA_VISIBLE_DEVICES"] = "7,8"
import torch
from torch.utils.data import (SequentialSampler)
import numpy as np
import random

from collections import OrderedDict
from nlgeval import NLGEval
import time
import argparse
from modules.tokenization import BertTokenizer
from modules.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from modules.modeling_UFS import UniVL

from modules.optimization import BertAdam
from modules.beam import Beam
from torch.utils.data import DataLoader
from dataloaders.dataloader_youcook_caption import Youcook_Caption_DataLoader
from dataloaders.dataloader_msrvtt_caption_UFS import MSRVTT_Caption_DataLoader,MSRVTT_Caption_DataLoader_Swin
from util import get_logger
from torch.utils.tensorboard import SummaryWriter
tbwriter = SummaryWriter('/home/gujiayang/workspace/videocaption/UniVL/log_3')
test_writer = SummaryWriter('/home/gujiayang/workspace/videocaption/UniVL/result_logs')
torch.distributed.init_process_group(backend="nccl")

global logger

def get_args(description='UniVL on Caption Task'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--do_pretrain", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--do_swin", action='store_true', help="Whether to run swin backbone.")

    parser.add_argument('--train_csv', type=str, default='data/youcookii_singlef_train.csv', help='')
    parser.add_argument('--val_csv', type=str, default='data/youcookii_singlef_val.csv', help='')
    parser.add_argument('--data_path', type=str, default='data/youcookii_caption_transcript.pickle',
                        help='caption and transcription pickle file path')
    parser.add_argument('--features_path', type=str, default='data/youcookii_videos_feature.pickle',
                        help='feature path for 2D features')

    parser.add_argument('--num_thread_reader', type=int, default=1, help='')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--batch_size_val', type=int, default=3500, help='batch size eval')
    parser.add_argument('--lr_decay', type=float, default=0.9, help='Learning rate exp epoch decay')
    parser.add_argument('--n_display', type=int, default=1, help='Information display frequence')
    parser.add_argument('--video_dim', type=int, default=1024, help='video feature dimension')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--max_words', type=int, default=20, help='')
    parser.add_argument('--max_frames', type=int, default=100, help='')
    parser.add_argument('--feature_framerate', type=int, default=1, help='')
    parser.add_argument('--min_time', type=float, default=5.0, help='Gather small clips')
    parser.add_argument('--margin', type=float, default=0.1, help='margin for loss')
    parser.add_argument('--hard_negative_rate', type=float, default=0.5, help='rate of intra negative sample')
    parser.add_argument('--negative_weighting', type=int, default=1, help='Weight the loss for intra negative')
    parser.add_argument('--n_pair', type=int, default=1, help='Num of pair to output from data loader')

    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--bert_model", default="bert-base-uncased", type=str, required=True, help="Bert pre-trained model")
    parser.add_argument("--visual_model", default="visual-base", type=str, required=False, help="Visual module")
    parser.add_argument("--cross_model", default="cross-base", type=str, required=False, help="Cross module")
    parser.add_argument("--decoder_model", default="decoder-base", type=str, required=False, help="Decoder module")
    parser.add_argument("--init_model", default=None, type=str, required=False, help="Initial model.")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--n_gpu', type=int, default=8, help="Changed in the execute process.")

    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    parser.add_argument("--task_type", default="caption", type=str, help="Point the task `caption` to finetune.")
    parser.add_argument("--datatype", default="youcook", type=str, help="Point the dataset `youcook` to finetune.")

    parser.add_argument("--world_size", default=0, type=int, help="distribted training")
    parser.add_argument("--local_rank", default=0, type=int, help="distribted training")
    parser.add_argument('--coef_lr', type=float, default=0.1, help='coefficient for bert branch.')
    parser.add_argument('--use_mil', action='store_true', help="Whether use MIL as Miech et. al. (2020).")
    parser.add_argument('--sampled_use_mil', action='store_true', help="Whether use MIL, has a high priority than use_mil.")

    parser.add_argument('--text_num_hidden_layers', type=int, default=12, help="Layer NO. of text.")
    parser.add_argument('--visual_num_hidden_layers', type=int, default=6, help="Layer NO. of visual.")
    parser.add_argument('--cross_num_hidden_layers', type=int, default=2, help="Layer NO. of cross.")
    parser.add_argument('--decoder_num_hidden_layers', type=int, default=3, help="Layer NO. of decoder.")

    parser.add_argument('--stage_two', action='store_true', help="Whether training with decoder.")
    parser.add_argument('--clip_num', type=int, default=8, help="siamese sampling clip num")
    parser.add_argument('--frame_num', type=int, default=8, help="siamese sampling clip num")
    args = parser.parse_args()

    # Check paramenters
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    args.batch_size = int(args.batch_size / args.gradient_accumulation_steps)

    return args

def set_seed_logger(args):
    global logger
    # predefining random initial seeds
    # To generate random sampling index,don't need to set all seeds
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    # np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    world_size = torch.distributed.get_world_size()
    torch.cuda.set_device(args.local_rank)
    args.world_size = world_size
    args.output_dir = args.output_dir+'_'+str(time.ctime())
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    logger = get_logger(os.path.join(args.output_dir, "log.txt"))

    if args.local_rank == 0:
        logger.info("Effective parameters:")
        for key in sorted(args.__dict__):
            logger.info("  <<< {}: {}".format(key, args.__dict__[key]))

    return args

def init_device(args, local_rank):
    global logger

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", local_rank)

    n_gpu = torch.cuda.device_count()
    logger.info("device: {} n_gpu: {}".format(device, n_gpu))
    args.n_gpu = n_gpu

    if args.batch_size % args.n_gpu != 0 or args.batch_size_val % args.n_gpu != 0:
        raise ValueError("Invalid batch_size/batch_size_val and n_gpu parameter: {}%{} and {}%{}, should be == 0".format(
            args.batch_size, args.n_gpu, args.batch_size_val, args.n_gpu))

    return device, n_gpu

def init_model(args, device, n_gpu, local_rank):

    if args.init_model:
        # model_state_dict = torch.load('/home/gujiayang/workspace/videocaption/UniVL/_snapshot/ckpt_video-swin.pt', map_location='cpu')
        # old_pretrain_dict = torch.load(args.init_model, map_location='cpu')
        # old_pretrain_dict = {k: v for k, v in old_pretrain_dict.items() if 'decoder' in k or 'bert' in k}
        # model_state_dict = torch.load("/home/gujiayang/workspace/videocaption/pytorch_violet/_snapshot/ckpt_violet_pretrain.pt", map_location='cpu')
        # model_state_dict = model_state_dict.update(trsfr_swin_state_dict)
        # model_state_dict = {k: v for k, v in model_state_dict.items() if 'trsfr' in k}
        # model_state_dict.update(old_pretrain_dict)
        # decoder state_dict
        model_state_dict =torch.load("/home/gujiayang/workspace/videocaption/ECCV2022_submission_100/checkpoint/UFSSemantics.ckpt", map_location='cpu')['state_dict']
        
        # model_state_dict =torch.load("/home/gujiayang/data/model/UniVL/siamese_sampling/swin+cross+transformerDec/pytorch_model.bin.14", map_location='cpu')

        # 1. filter out unnecessary keys
        # model_state_dict = {k: v for k, v in model_state_dict.items() if 'encoder' in k}
        # 2. overwrite entries in the existing state dict
        # model_dict.update(pretrained_dict)
        # model_state_dict.update(pretrained_dict)
        
        # swin_dict = torch.load('/home/gujiayang/workspace/videocaption/UniVL/_snapshot/ckpt_video-swin.pt', map_location='cpu')

        # model_state_dict.update(swin_dict)
        
        
       
        # model_state_dict = dict(model_state_dict.item()+trsfr_swin_state_dict.item())
        
    else:
        model_state_dict = None
    
    # Prepare model
    cache_dir = "/home/gujiayang/workspace/videocaption/UniVL/cache_dir"    
    model = UniVL.from_pretrained(args.bert_model, args.visual_model, args.cross_model, args.decoder_model,
                                   cache_dir=cache_dir, state_dict=model_state_dict, task_config=args)
    

    model.to(device)
    

    return model

def prep_optimizer(args, model, num_train_optimization_steps, device, n_gpu, local_rank, coef_lr=1.):

    if hasattr(model, 'module'):
        model = model.module

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    # no_decay_param_tp = [(n, p) for n, p in param_optimizer if not any(nd in n for nd in no_decay)]
    # decay_param_tp = [(n, p) for n, p in param_optimizer if any(nd in n for nd in no_decay)]

    # no_decay_bert_param_tp = [(n, p) for n, p in no_decay_param_tp if "bert." in n]
    # no_decay_nobert_param_tp = [(n, p) for n, p in no_decay_param_tp if "bert." not in n]

    # decay_bert_param_tp = [(n, p) for n, p in decay_param_tp if "bert." in n]
    # decay_nobert_param_tp = [(n, p) for n, p in decay_param_tp if "bert." not in n]
    
    # decay_swin_trsfr_param_tp = [(n, p) for n, p in decay_nobert_param_tp if  "encoder.model" in n  or "trsfr." in n]
    # no_decay_swin_trsfr_param_tp = [(n, p) for n, p in no_decay_nobert_param_tp if  "encoder.model" in n or "trsfr." in n]
    
    # no_decay_nobert_param_tp_1 = [(n, p) for n, p in no_decay_param_tp if "bert." not in n and "encoder.model" not in n and "trsfr." not in n]
    # decay_nobert_param_tp_1 = [(n, p) for n, p in decay_param_tp if "bert." not in n and "encoder.model" not in n and "trsfr." not in n]
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in no_decay_bert_param_tp], 'weight_decay': 0.01, 'lr': args.lr * coef_lr},
    #     {'params': [p for n, p in no_decay_nobert_param_tp_1], 'weight_decay': 0.01},
    #     {'params': [p for n, p in decay_bert_param_tp], 'weight_decay': 0.0, 'lr': args.lr * coef_lr},
    #     {'params': [p for n, p in decay_nobert_param_tp_1], 'weight_decay': 0.0},
    #     {'params': [p for n, p in decay_swin_trsfr_param_tp], 'weight_decay': 1e-3,'lr':3e-6},
    #     {'params': [p for n, p in no_decay_swin_trsfr_param_tp], 'weight_decay': 1e-3,'lr':3e-6}
        
    # ]
    
    no_decay_param_tp = [(n, p) for n, p in param_optimizer if not any(nd in n for nd in no_decay)]
    decay_param_tp = [(n, p) for n, p in param_optimizer if any(nd in n for nd in no_decay)]

    no_decay_decoder_param_tp = [(n, p) for n, p in no_decay_param_tp if "decoder." in n]
    no_decay_no_decoder_param_tp = [(n, p) for n, p in no_decay_param_tp if "decoder." not in n]

    decay_decoder_param_tp = [(n, p) for n, p in decay_param_tp if "decoder." in n]
    decay_no_decoder_param_tp = [(n, p) for n, p in decay_param_tp if "decoder." not in n]
    
   
    
    
    optimizer_grouped_parameters = [
        {'params': [p for n, p in no_decay_decoder_param_tp], 'weight_decay': 0, 'lr': 0.00001},
        {'params': [p for n, p in no_decay_no_decoder_param_tp], 'weight_decay': 0 , 'lr': 0.00001},
        {'params': [p for n, p in decay_decoder_param_tp], 'weight_decay': 0.0, 'lr': 0.00001},
        {'params': [p for n, p in decay_no_decoder_param_tp], 'weight_decay': 0.0,'lr': 0.00001},
        
    ]
    
    scheduler = None
    scheduler = None
    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.lr, warmup=args.warmup_proportion,
                         schedule='warmup_linear', t_total=num_train_optimization_steps, weight_decay=0.01,
                         max_grad_norm=1.0)
    


    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                      output_device=local_rank, find_unused_parameters=True)
    # optimizer = torch.optim.AdamW(model.named_parameters(), lr=0.00001)
    return optimizer, scheduler, model

def dataloader_youcook_train(args, tokenizer):
    youcook_dataset = Youcook_Caption_DataLoader(
        csv=args.train_csv,
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(youcook_dataset)
    dataloader = DataLoader(
        youcook_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(youcook_dataset), train_sampler

def dataloader_youcook_test(args, tokenizer):
    youcook_testset = Youcook_Caption_DataLoader(
        csv=args.val_csv,
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
    )

    test_sampler = SequentialSampler(youcook_testset)
    dataloader_youcook = DataLoader(
        youcook_testset,
        sampler=test_sampler,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        pin_memory=False,
    )

    if args.local_rank == 0:
        logger.info('YoucookII validation pairs: {}'.format(len(youcook_testset)))
    return dataloader_youcook, len(youcook_testset)

def dataloader_msrvtt_train(args, tokenizer):
    msrvtt_dataset = MSRVTT_Caption_DataLoader(
        csv_path=args.train_csv,
        json_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        split_type="train"
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(msrvtt_dataset)
    dataloader = DataLoader(
        msrvtt_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(msrvtt_dataset), train_sampler
def dataloader_msrvtt_train_swin(args, tokenizer):

    msrvtt_dataset_swin = MSRVTT_Caption_DataLoader_Swin(
        csv_path=args.train_csv,
        json_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        split_type="train",
        config = "/home/gujiayang/workspace/videocaption/UniVL/config/swin_base_bert.py"
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(msrvtt_dataset_swin)
    
    dataloader = DataLoader(
        msrvtt_dataset_swin,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(msrvtt_dataset_swin), train_sampler

def dataloader_msrvtt_test(args, tokenizer, split_type="test",):
    msrvtt_testset = MSRVTT_Caption_DataLoader(
        csv_path=args.val_csv,
        json_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        split_type=split_type,
    )
    test_sampler = SequentialSampler(msrvtt_testset)
    dataloader_msrvtt = DataLoader(
        msrvtt_testset,
        sampler=test_sampler,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        drop_last=False,
    )
    return dataloader_msrvtt, len(msrvtt_testset)
def dataloader_msrvtt_test_swin(args, tokenizer, split_type="test",):
    msrvtt_testset = MSRVTT_Caption_DataLoader_Swin(
        csv_path=args.val_csv,
        json_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        split_type=split_type,
        config = "/home/gujiayang/workspace/videocaption/UniVL/config/swin_base_bert.py"
    )
    
    test_sampler = SequentialSampler(msrvtt_testset)
    # test_sampler = torch.utils.data.distributed.DistributedSampler(msrvtt_testset)
    
    dataloader_msrvtt = DataLoader(
        msrvtt_testset,
        sampler=test_sampler,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        drop_last=False,
    )
    return dataloader_msrvtt, len(msrvtt_testset)

def convert_state_dict_type(state_dict, ttype=torch.FloatTensor):
    if isinstance(state_dict, dict):
        cpu_dict = OrderedDict()
        for k, v in state_dict.items():
            cpu_dict[k] = convert_state_dict_type(v)
        return cpu_dict
    elif isinstance(state_dict, list):
        return [convert_state_dict_type(v) for v in state_dict]
    elif torch.is_tensor(state_dict):
        return state_dict.type(ttype)
    else:
        return state_dict

def save_model(epoch, args, model, type_name=""):
    # Only save the model it-self
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(
        args.output_dir, "pytorch_model.bin.{}{}".format("" if type_name=="" else type_name+".", epoch))
    torch.save(model_to_save.state_dict(), output_model_file)
    logger.info("Model saved to %s", output_model_file)
    return output_model_file

def load_model(epoch, args, n_gpu, device, model_file=None):
    if model_file is None or len(model_file) == 0:
        model_file = os.path.join(args.output_dir, "pytorch_model.bin.{}".format(epoch))
    if os.path.exists(model_file):
        model_state_dict = torch.load(model_file, map_location='cpu')

        
        bert_dict = torch.load("/home/gujiayang/data/model/UniVL/siamese_sampling/BEST_after_3*8_loss2anchor/pytorch_model.bin.4",map_location='cpu')
        bert_dict = {k: v for k, v in bert_dict.items() if 'bert' in k}
        
        # model_state_dict.update(cross_dict)
        model_state_dict.update(bert_dict)
        if args.local_rank == 0:
            logger.info("Model loaded from %s", model_file)
        # Prepare model
        cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
        model = UniVL.from_pretrained(args.bert_model, args.visual_model, args.cross_model, args.decoder_model,
                                       cache_dir=cache_dir, state_dict=model_state_dict, task_config=args)

        model.to(device)
    else:
        model = None
    return model
def padding_video(video ,max_length):
    # video = [ clip_num , frm_per_clip , hidden_feature ]
    clip_num = video.size(0)
    frm_per_clip = video.size(1)
    # padding_video = [8 , 48 , 1024]
    padding_video = torch.zeros(clip_num,max_length,video.size(2))
    for i in range(video.size(0)):
        padding_video[i][:frm_per_clip]=video[i]
    return padding_video
    
def getSiameseClips(video,siamese_video,max_frames):
    siamese_output = siamese_video.permute(1,0,2,3)
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
        
        single_batch_video = video[batch_num][0]
        # gathered_batch_video = [clip_num , 5 , 1024]
        gathered_batch_video = single_batch_video.unsqueeze(1).expand(48,5,1024).gather(dim = 0 ,index=sia_idx.unsqueeze(2).expand(8,5,1024))
        # gathered_batch_video = [clip_num, 5 + padding=48 , 1024] == [ 8 , 48 , 1024 ] 
        # need to padding second dim to match frm_max_length
        # gathered_batch_video_padding = padding_video(gathered_batch_video,max_frames)
        # tempoutput = [bz , 8 , 48 , 1024]
        siamese_output[batch_num] = gathered_batch_video
    
    
    siamese_video = siamese_output.permute(1,0,2,3)
    
    # siamese_video = [8 , bz , 48 , 1024]
    # output = [8*bz , 5, 1024]
    siamese_video = siamese_video.reshape(-1,siamese_video.size(-2),siamese_video.size(-1))
    return siamese_video
    
def train_epoch(epoch, args, model, train_dataloader, tokenizer, device, n_gpu, optimizer, scheduler,
                global_step, nlgEvalObj=None, local_rank=0):
    global logger
    torch.cuda.empty_cache()
    model.train()
    log_step = args.n_display
    start_time = time.time()
    starttime = time.perf_counter()
    total_loss = 0

    for step, batch in enumerate(train_dataloader):
        # if n_gpu == 1:
        #     # multi-gpu does scattering it-self
        #     batch = tuple(t.to(device) for t in batch)·
        batch = tuple(t.to(device=device, non_blocking=True) for t in batch)

        input_ids, input_mask, segment_ids, video, video_mask, \
        pairs_masked_text, pairs_token_labels, masked_video, video_labels_index,\
        pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids  = batch
        # siamese_video = [clip , batch , frame_num  = 5 , hidden_features(1024)]
        siamese_video = torch.zeros(8,video.size(0),5,video.size(3))
        
        # siamese_video = getSiameseClips(video,siamese_video,args.max_frames)
        # print(f'loading:{time.perf_counter() - starttime:.8f}s')
        starttime = time.perf_counter()
        # video = [bz,48,3,224,224]
        output_trigger = False
        if global_step >=100 and global_step%50 ==0:
            output_trigger =True
        loss,siamese_loss = model(input_ids, segment_ids, input_mask, video, siamese_video,video_mask,
                     pairs_masked_text=pairs_masked_text, pairs_token_labels=pairs_token_labels,
                     masked_video=masked_video, video_labels_index=video_labels_index,
                     input_caption_ids=pairs_input_caption_ids, decoder_mask=pairs_decoder_mask,
                     output_caption_ids=pairs_output_caption_ids,siamese_trigger = False,swin_trigger = True,tokenizer = tokenizer,output_trigger = output_trigger)
        tbwriter.add_scalar('total loss', loss, global_step=global_step, walltime=None)
        tbwriter.add_scalar('siamese loss', siamese_loss, global_step=global_step, walltime=None)
        # print(f'model sum:{time.perf_counter() - starttime:.8f}s')
        starttime = time.perf_counter()
        if n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        loss.backward()
        # print(f'backward:{time.perf_counter() - starttime:.8f}s')
        total_loss += float(loss)
        if (step + 1) % args.gradient_accumulation_steps == 0:

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if scheduler is not None:
                scheduler.step()  # Update learning rate schedule

            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
            if global_step % log_step == 0 and local_rank == 0:
                logger.info("Epoch: %d/%s, Step: %d/%d, Lr: %s, Loss: %f, Time/step: %f", epoch + 1,
                            args.epochs, step + 1,
                            len(train_dataloader), "-".join([str('%.6f'%itm) for itm in sorted(list(set(optimizer.get_lr())))]),
                            float(loss),
                            (time.time() - start_time) / (log_step * args.gradient_accumulation_steps))
                start_time = time.time()

    total_loss = total_loss / len(train_dataloader)
    return total_loss, global_step

def beam_decode_step(decoder, inst_dec_beams, len_dec_seq,
                     inst_idx_to_position_map, n_bm, device, input_tuples, decoder_length=None):

    assert isinstance(input_tuples, tuple)

    ''' Decode and update beam status, and then return active beam idx'''
    def prepare_beam_dec_seq(inst_dec_beams, len_dec_seq):
        dec_partial_seq = [b.get_current_state() for b in inst_dec_beams if not b.done]
        dec_partial_seq = torch.stack(dec_partial_seq).to(device)
        dec_partial_seq = dec_partial_seq.view(-1, len_dec_seq)
        return dec_partial_seq

    def predict_word(next_decoder_ids, n_active_inst, n_bm, device, input_tuples):
        sequence_output_rpt, visual_output_rpt, input_ids_rpt, input_mask_rpt, video_mask_rpt = input_tuples
        next_decoder_mask = torch.ones(next_decoder_ids.size(), dtype=torch.uint8).to(device)

        # dec_output = decoder(sequence_output_rpt, visual_output_rpt, input_ids_rpt, input_mask_rpt,
        #                      video_mask_rpt, next_decoder_ids, next_decoder_mask, shaped=True, get_logits=True)
        dec_output = decoder(input_ids_rpt, encoder_outs=visual_output_rpt, answer_mask=next_decoder_mask, encoder_mask=video_mask_rpt)
        dec_output = dec_output[:, -1, :]
        word_prob = torch.nn.functional.log_softmax(dec_output, dim=1)
        word_prob = word_prob.view(n_active_inst, n_bm, -1)
        return word_prob

    def collect_active_inst_idx_list(inst_beams, word_prob, inst_idx_to_position_map, decoder_length=None):
        active_inst_idx_list = []
        for inst_idx, inst_position in inst_idx_to_position_map.items():
            if decoder_length is None:
                is_inst_complete = inst_beams[inst_idx].advance(word_prob[inst_position])
            else:
                is_inst_complete = inst_beams[inst_idx].advance(word_prob[inst_position], word_length=decoder_length[inst_idx])
            if not is_inst_complete:
                active_inst_idx_list += [inst_idx]

        return active_inst_idx_list

    n_active_inst = len(inst_idx_to_position_map)
    dec_seq = prepare_beam_dec_seq(inst_dec_beams, len_dec_seq)
    word_prob = predict_word(dec_seq, n_active_inst, n_bm, device, input_tuples)

    # Update the beam with predicted word prob information and collect incomplete instances
    active_inst_idx_list = collect_active_inst_idx_list(inst_dec_beams, word_prob, inst_idx_to_position_map,
                                                        decoder_length=decoder_length)

    return active_inst_idx_list



def UFS_decode_step(decoder, input_video):
    ''' Decode and update beam status, and then return active beam idx'''
    dec_output = decoder(input_video)
    return dec_output


# >----------------------------------------

def eval_epoch(args, model, test_dataloader, tokenizer, device, n_gpu, nlgEvalObj=None, test_set=None):

    if hasattr(model, 'module'):
        model = model.module.to(device)

    # if model._stage_one:
    #     return 0.

    all_result_lists = []
    all_caption_lists = []
    model.eval()
    total_text_generated = []
    for batch in test_dataloader:
        batch = tuple(t.to(device, non_blocking=True) for t in batch)

        input_ids, input_mask, segment_ids, video, video_mask, \
        pairs_masked_text, pairs_token_labels, masked_video, video_labels_index, \
        pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids = batch

        with torch.no_grad():
            # -- Repeat data for beam search
            decoder = model.decoder_caption_with_cross
            result_list = UFS_decode_step(decoder,video)
            generate_text = result_list.cpu().numpy().tolist()

            generate_converted = model.tokenizer_lm.batch_decode(generate_text, skip_special_tokens=True)
            
            for di in range(generate_converted.__len__()):
                print(generate_converted[di])
                total_text_generated.append(generate_converted[di])
                
            pairs_output_caption_ids = pairs_output_caption_ids.view(-1, pairs_output_caption_ids.shape[-1])
            caption_list = pairs_output_caption_ids.cpu().detach().numpy()
            
        

            for re_idx, re_list in enumerate(caption_list):
                decode_text_list = tokenizer.convert_ids_to_tokens(re_list)
                if "[SEP]" in decode_text_list:
                    SEP_index = decode_text_list.index("[SEP]")
                    decode_text_list = decode_text_list[:SEP_index]
                if "[PAD]" in decode_text_list:
                    PAD_index = decode_text_list.index("[PAD]")
                    decode_text_list = decode_text_list[:PAD_index]
                decode_text = ' '.join(decode_text_list)
                decode_text = decode_text.replace(" ##", "").strip("##").strip()
                all_caption_lists.append(decode_text)

    
    

    ref_path = os.path.join(args.output_dir, "ref.txt")
    with open(ref_path, "w", encoding='utf-8') as writer:
        for ground_txt in all_caption_lists:
            writer.write(ground_txt + "\n")

    if args.datatype == "msrvtt":
        all_caption_lists = []
        sentences_dict = test_dataloader.dataset.sentences_dict
        video_sentences_dict = test_dataloader.dataset.video_sentences_dict
        for idx in range(len(sentences_dict)):
            video_id, _ = sentences_dict[idx]
            sentences = video_sentences_dict[video_id]
            all_caption_lists.append(sentences)
        all_caption_lists = [list(itms) for itms in zip(*all_caption_lists)]
    else:
        all_caption_lists = [all_caption_lists]

    # Evaluate
    metrics_nlg = nlgEvalObj.compute_metrics(ref_list=all_caption_lists, hyp_list=total_text_generated)

    logger.info(">>>  BLEU_1: {:.4f}, BLEU_2: {:.4f}, BLEU_3: {:.4f}, BLEU_4: {:.4f}".
                format(metrics_nlg["Bleu_1"], metrics_nlg["Bleu_2"], metrics_nlg["Bleu_3"], metrics_nlg["Bleu_4"]))
    logger.info(">>>  METEOR: {:.4f}, ROUGE_L: {:.4f}, CIDEr: {:.4f}".format(metrics_nlg["METEOR"], metrics_nlg["ROUGE_L"], metrics_nlg["CIDEr"]))
    
   
    Bleu_4 = metrics_nlg["Bleu_4"]
    return Bleu_4,metrics_nlg

DATALOADER_DICT = {}
DATALOADER_DICT["youcook"] = {"train":dataloader_youcook_train, "val":dataloader_youcook_test}
DATALOADER_DICT["msrvtt"] = {"train":dataloader_msrvtt_train, "val":dataloader_msrvtt_test,"train_swin":dataloader_msrvtt_train_swin,"val_swin":dataloader_msrvtt_test_swin}

def main():
    global logger
    args = get_args()
    args = set_seed_logger(args)
    device, n_gpu = init_device(args, args.local_rank)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    

    assert args.task_type == "caption"
    nlgEvalObj = NLGEval(no_overlap=False, no_skipthoughts=True, no_glove=True, metrics_to_omit=None)

    assert args.datatype in DATALOADER_DICT
    
    if args.local_rank == 0:
        logger.info("***** Running test *****")
        # logger.info("  Num examples = %d", test_length)
        logger.info("  Batch size = %d", args.batch_size_val)
        # logger.info("  Num steps = %d", len(test_dataloader))

    if args.do_train and args.do_swin==False:
        train_dataloader, train_length, train_sampler = DATALOADER_DICT[args.datatype]["train"](args, tokenizer)
        num_train_optimization_steps = (int(len(train_dataloader) + args.gradient_accumulation_steps - 1)
                                        / args.gradient_accumulation_steps) * args.epochs
        coef_lr = args.coef_lr
        if args.init_model:
            model = init_model(args, device, n_gpu, args.local_rank)
            coef_lr = 1.0
        optimizer, scheduler, model = prep_optimizer(args, model, num_train_optimization_steps, device, n_gpu, args.local_rank, coef_lr=coef_lr)

        if args.local_rank == 0:
            logger.info("***** Running training *****")
            logger.info("  Num examples = %d", train_length)
            logger.info("  Batch size = %d", args.batch_size)
            logger.info("  Num steps = %d", num_train_optimization_steps * args.gradient_accumulation_steps)

        best_score = 0.00001
        best_output_model_file = None
        global_step = 0
        for epoch in range(args.epochs):
            train_sampler.set_epoch(epoch)

            tr_loss, global_step = train_epoch(epoch, args, model, train_dataloader, tokenizer, device, n_gpu, optimizer,
                                               scheduler, global_step, nlgEvalObj=nlgEvalObj, local_rank=args.local_rank)

            if args.local_rank == 0:
                logger.info("Epoch %d/%s Finished, Train Loss: %f", epoch + 1, args.epochs, tr_loss)
                output_model_file = save_model(epoch, args, model, type_name="")
                if epoch >= 1:
                    if args.do_swin == True:
                        test_dataloader, test_length = DATALOADER_DICT[args.datatype]["val_swin"](args, tokenizer)
                    else:
                        test_dataloader, test_length = DATALOADER_DICT[args.datatype]["val"](args, tokenizer)
                    logger.info("***** Running test *****")
                    logger.info("  Num examples = %d", test_length)
                    logger.info("  Batch size = %d", args.batch_size_val)
                    logger.info("  Num steps = %d", len(test_dataloader))
                    Bleu_4 = eval_epoch(args, model, test_dataloader, tokenizer, device, n_gpu, nlgEvalObj=nlgEvalObj)
                    if best_score <= Bleu_4:
                        best_score = Bleu_4
                        best_output_model_file = output_model_file
                    logger.info("The best model is: {}, the Bleu_4 is: {:.4f}".format(best_output_model_file, best_score))
                else:
                    logger.warning("Skip the evaluation after {}-th epoch.".format(epoch+1))

        if args.local_rank == 0:
            if args.do_swin == True:
                test_dataloader, test_length = DATALOADER_DICT[args.datatype]["val_swin"](args, tokenizer)
            else:
                test_dataloader, test_length = DATALOADER_DICT[args.datatype]["val"](args, tokenizer)
            logger.info("***** Running test *****")
            logger.info("  Num examples = %d", test_length)
            logger.info("  Batch size = %d", args.batch_size_val)
            logger.info("  Num steps = %d", len(test_dataloader))
            model = load_model(-1, args, n_gpu, device, model_file=best_output_model_file)
            eval_epoch(args, model, test_dataloader, tokenizer, device, n_gpu, nlgEvalObj=nlgEvalObj)
    elif args.do_train:
        train_dataloader, train_length, train_sampler = DATALOADER_DICT[args.datatype]["train_swin"](args, tokenizer)
        num_train_optimization_steps = (int(len(train_dataloader) + args.gradient_accumulation_steps - 1)
                                        / args.gradient_accumulation_steps) * args.epochs
        coef_lr = args.coef_lr
        if args.init_model:
            model = init_model(args, device, n_gpu, args.local_rank)
            coef_lr = 1.0
        optimizer, scheduler, model = prep_optimizer(args, model, num_train_optimization_steps, device, n_gpu, args.local_rank, coef_lr=coef_lr)

        if args.local_rank == 0:
            logger.info("***** Running training *****")
            logger.info("  Num examples = %d", train_length)
            logger.info("  Batch size = %d", args.batch_size)
            logger.info("  Num steps = %d", num_train_optimization_steps * args.gradient_accumulation_steps)

        best_score = 0.00001
        best_output_model_file = None
        global_step = 0
        for epoch in range(args.epochs):
            train_sampler.set_epoch(epoch)

            tr_loss, global_step = train_epoch(epoch, args, model, train_dataloader, tokenizer, device, n_gpu, optimizer,
                                               scheduler, global_step, nlgEvalObj=nlgEvalObj, local_rank=args.local_rank)
            # tbwriter.add_scalar('total loss', tr_loss, global_step=global_step, walltime=None)

            if args.local_rank == 0:
                logger.info("Epoch %d/%s Finished, Train Loss: %f", epoch + 1, args.epochs, tr_loss)
                output_model_file = save_model(epoch, args, model, type_name="")
                if epoch >= 30:
                    if args.do_swin == True:
                        test_dataloader, test_length = DATALOADER_DICT[args.datatype]["val_swin"](args, tokenizer)
                    else:
                        test_dataloader, test_length = DATALOADER_DICT[args.datatype]["val"](args, tokenizer)
                    logger.info("***** Running test *****")
                    logger.info("  Num examples = %d", test_length)
                    logger.info("  Batch size = %d", args.batch_size_val)
                    logger.info("  Num steps = %d", len(test_dataloader))
                    Bleu_4,cider= eval_epoch(args, model, test_dataloader, tokenizer, device, n_gpu, nlgEvalObj=nlgEvalObj)
                    # tbwriter.add_scalar('cider %d', cider, global_step=global_step, walltime=None)
                    if best_score <= Bleu_4:
                        best_score = Bleu_4
                        best_output_model_file = output_model_file
                    logger.info("The best model is: {}, the Bleu_4 is: {:.4f}".format(best_output_model_file, best_score))
                else:
                    logger.warning("Skip the evaluation after {}-th epoch.".format(epoch+1))

        if args.local_rank == 0:
            if args.do_swin == True:
                test_dataloader, test_length = DATALOADER_DICT[args.datatype]["val_swin"](args, tokenizer)
            else:
                test_dataloader, test_length = DATALOADER_DICT[args.datatype]["val"](args, tokenizer)
            logger.info("***** Running test *****")
            logger.info("  Num examples = %d", test_length)
            logger.info("  Batch size = %d", args.batch_size_val)
            logger.info("  Num steps = %d", len(test_dataloader))
            model = load_model(-1, args, n_gpu, device, model_file=best_output_model_file)
            eval_epoch(args, model, test_dataloader, tokenizer, device, n_gpu, nlgEvalObj=nlgEvalObj)
    elif args.do_eval:
        if args.local_rank == 0:
            if args.do_swin == True:
                test_dataloader, test_length = DATALOADER_DICT[args.datatype]["val_swin"](args, tokenizer)
            else:
                test_dataloader, test_length = DATALOADER_DICT[args.datatype]["val"](args, tokenizer)
            logger.info("***** Running test *****")
            logger.info("  Num examples = %d", test_length)
            logger.info("  Batch size = %d", args.batch_size_val)
            logger.info("  Num steps = %d", len(test_dataloader))
            # model = load_model(-1, args, n_gpu, device, model_file="/home/gujiayang/data/model/UniVL/siamese_sampling/BEST_after_3*8_loss2anchor/pytorch_model.bin.4")
            # model = load_model(-1, args, n_gpu, device, model_file="/home/gujiayang/workspace/videocaption/ECCV2022_submission_100/checkpoint/UFSSemantics.ckpt")
            for i in range(4,6):
                model = load_model(-1, args, n_gpu, device, model_file="/home/gujiayang/data/model/UniVL/siamese_sampling/ckpt_msrvtt_caption_Sun Apr 10 11:13:29 2022/pytorch_model.bin."+str(i))
                model.encoder.eval()
                model.decoder.eval()
                # torch.cuda.set_device(args.local_rank)
                # device = torch.device("cuda", args.local_rank)
                # model.to(device)
                # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                #                                       output_device=args.local_rank, find_unused_parameters=True)
                _ , metric = eval_epoch(args, model, test_dataloader, tokenizer, device, n_gpu, nlgEvalObj=nlgEvalObj)
                
                test_writer.add_scalar('cider %d', metric["CIDEr"], global_step=i, walltime=None)
                test_writer.add_scalar('Bleu1 %d', metric["Bleu_1"], global_step=i, walltime=None)
                test_writer.add_scalar('Bleu2 %d', metric["Bleu_2"], global_step=i, walltime=None)
                test_writer.add_scalar('Bleu4 %d', metric["Bleu_4"], global_step=i, walltime=None)
                test_writer.add_scalar('Meteor %d', metric["METEOR"], global_step=i, walltime=None)
                test_writer.add_scalar('Rouge %d', metric["ROUGE_L"], global_step=i, walltime=None)


if __name__ == "__main__":
    main()