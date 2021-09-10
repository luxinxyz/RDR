# -*- coding: utf-8 -*-
import torch
from torch.utils.data import DataLoader
from od.inputters.dataset import DoubanDataset


def build_dist_loaders(args, tokenizer, logger):
    logger.info("Build train and validation dataloaders")

    logger.info("Train_Path: %s" % args.train_path)
    logger.info("Valid_Path: %s" % args.valid_path)
    train_dataset = DoubanDataset(tokenizer, senti=args.senti, data_path=args.train_path)
    valid_dataset = DoubanDataset(tokenizer, senti=args.senti, data_path=args.valid_path)
    logger.info("Train Num = {}, Valid Num = {}".format(len(train_dataset), len(valid_dataset)))
    logger.info("=====Train Example=====")
    for i in range(5):
        logger.info(train_dataset[i])
    logger.info("=====Valid Example=====")
    for i in range(5):
        logger.info(valid_dataset[i])
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset) if args.distributed else None
    train_loader = DataLoader(train_dataset,
                              collate_fn=train_dataset.collate,
                              pin_memory=(args.device == "cuda"),
                              num_workers=args.num_workers,
                              sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              shuffle=(not args.distributed))
    valid_loader = DataLoader(valid_dataset,
                              collate_fn=valid_dataset.collate,
                              pin_memory=(args.device == "cuda"),
                              num_workers=args.num_workers,
                              sampler=valid_sampler,
                              batch_size=args.valid_batch_size,
                              shuffle=False)
    return train_loader, valid_loader, train_sampler, valid_sampler
