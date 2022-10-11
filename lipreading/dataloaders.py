from functools import partial

import numpy as np
import random
import torch
from torch.utils.data import DataLoader

from lipreading.dataset import MyDataset, pad_packed_collate
from lipreading.preprocess import *
from lipreading.utils import SAMPLERS
from mmcv.utils import build_from_cfg


def build_sampler(cfg, default_args=None):
    if cfg is None:
        return None
    else:
        return build_from_cfg(cfg, SAMPLERS, default_args=default_args)


def get_preprocessing_pipelines(modality):
    # -- preprocess for the video stream
    preprocessing = {}
    # -- LRW config
    if modality == 'video':
        crop_size = (88, 88)
        (mean, std) = (0.421, 0.165)
        preprocessing['train'] = Compose([
            # RgbToGray(),
            Normalize(0.0, 255.0),
            RandomCrop(crop_size),
            HorizontalFlip(0.5),
            Normalize(mean, std),
            TimeMask(T=0.6 * 25, n_mask=1)
        ])

        preprocessing['val'] = Compose([
            # RgbToGray(),
            Normalize(0.0, 255.0),
            CenterCrop(crop_size),
            Normalize(mean, std)
        ])

        preprocessing['test'] = preprocessing['val']

    elif modality == 'audio':

        preprocessing['train'] = Compose([
            AddNoise(noise=np.load('./data/babbleNoise_resample_16K.npy')),
            NormalizeUtterance()
        ])

        preprocessing['val'] = NormalizeUtterance()

        preprocessing['test'] = NormalizeUtterance()

    return preprocessing


def worker_init_fn(worker_id, num_workers, rank, seed):
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def get_data_loaders(args):
    preprocessing = get_preprocessing_pipelines(args.modality)

    # create dataset object for each partition
    partitions = ['test'] if args.test else ['train', 'val', 'test']
    dsets = {
        partition: MyDataset(
            modality=args.modality,
            data_partition=partition,
            data_dir=args.data_dir,
            label_fp=args.label_path,
            annonation_direc=args.annonation_direc,
            preprocessing_func=preprocessing[partition],
            data_suffix='.npz',
            use_boundary=args.use_boundary,
        )
        for partition in partitions
    }

    init_fn = partial(worker_init_fn,
                      num_workers=args.workers,
                      rank=args.rank,
                      seed=args.seed) if args.seed is not None else None

    dset_loaders = {}
    for key in partitions:
        if key == "train":
            sampler = build_sampler(
                dict(type='DistributedSampler',
                     dataset=dsets[key],
                     num_replicas=args.world_size,
                     rank=args.rank,
                     shuffle=True,
                     round_up=True,
                     seed=args.seed))
            dset_loaders[key] = DataLoader(dsets[key],
                                           batch_size=args.batch_size //
                                           args.world_size,
                                           sampler=sampler,
                                           num_workers=args.workers,
                                           collate_fn=pad_packed_collate,
                                           pin_memory=True,
                                           shuffle=False,
                                           worker_init_fn=init_fn)
        else:
            sampler = None
            dset_loaders[key] = DataLoader(dsets[key],
                                           batch_size=args.batch_size,
                                           sampler=sampler,
                                           num_workers=args.workers,
                                           collate_fn=pad_packed_collate,
                                           pin_memory=True,
                                           shuffle=False,
                                           drop_last=False,
                                           worker_init_fn=init_fn)


#     dset_loaders = {
#         x: torch.utils.data.DataLoader(dsets[x],
#                                        batch_size=args.batch_size,
#                                        shuffle=True,
#                                        collate_fn=pad_packed_collate,
#                                        pin_memory=True,
#                                        workers=args.workers,
#                                        worker_init_fn=np.random.seed(1))
#         for x in partitions
#     }
    return dset_loaders
