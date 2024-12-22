import os, random, json
from argparse import Namespace

import torch
import torch.utils.data as data_utils

from options import args
# with open("./config.json", "r") as f:
#     args = json.load(f)

# args = Namespace(**args)

from utils import *
from datasets.ml_1m import ML1MDataset
from datasets.base import AbstractDataset
from dataloaders.negative_samplers import negative_sampler_factory
from dataloaders.bert import BertDataloader, BertTrainDataset, BertEvalDataset
from models.bert import BERTModel
from trainers.bert import BERTTrainer

def main():
    export_root = setup_train(args)

    # Dataset
    dataset = ML1MDataset(args)
    _dataset = dataset.load_dataset()

    # Negative Sampler
    code = args.train_negative_sampler_code
    train_negative_sampler = negative_sampler_factory(code, _dataset['train'], _dataset['val'], _dataset['test'],
                                                      len(_dataset['umap']), len(_dataset['smap']),
                                                      args.train_negative_sample_size,
                                                      args.train_negative_sampling_seed,
                                                      dataset._get_preprocessed_folder_path())
    
    code = args.test_negative_sampler_code
    test_negative_sampler = negative_sampler_factory(code, _dataset['train'], _dataset['val'], _dataset['test'],
                                                     len(_dataset['umap']), len(_dataset['smap']),
                                                     args.test_negative_sample_size,
                                                     args.test_negative_sampling_seed,
                                                     dataset._get_preprocessed_folder_path())


    # DataLoader
    seed = args.dataloader_random_seed
    args.num_items = len(_dataset['smap'])

    train_dataset = BertTrainDataset(u2seq=_dataset['train'], 
                                     max_len=args.bert_max_len, 
                                     mask_prob=args.bert_mask_prob, 
                                     mask_token=len(_dataset['smap'])+1, 
                                     num_items=len(_dataset['smap']), 
                                     rng=random.Random(seed))
    eval_dataset = BertEvalDataset(u2seq=_dataset['train'], 
                                   u2answer=_dataset['val'], 
                                   max_len=args.bert_max_len,  
                                   mask_token=len(_dataset['smap'])+1,
                                   negative_samples=test_negative_sampler.get_negative_samples())
    test_dataset = BertEvalDataset(u2seq=_dataset['train'], 
                                   u2answer=_dataset['test'], 
                                   max_len=args.bert_max_len,  
                                   mask_token=len(_dataset['smap'])+1,
                                   negative_samples=test_negative_sampler.get_negative_samples())

    train_loader = data_utils.DataLoader(train_dataset, 
                                         batch_size=args.train_batch_size,
                                         shuffle=True, 
                                         pin_memory=True)    
    val_loader = data_utils.DataLoader(eval_dataset, batch_size=args.val_batch_size,
                                       shuffle=False, pin_memory=True)
    test_loader = data_utils.DataLoader(test_dataset, batch_size=args.test_batch_size,
                                       shuffle=False, pin_memory=True)

    # Train
    model = BERTModel(args)
    trainer = BERTTrainer(args, model, train_loader, val_loader, test_loader, export_root)

    trainer.train()

    # Test
    test_model = (input('Test model with test dataset? y/[n]: ') == 'y')
    if test_model:
        trainer.test()
    


if __name__ == '__main__':
    if args.mode == 'train':
        main()
    else:
        raise ValueError('Invalid mode')
