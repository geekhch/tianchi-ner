import os
from os.path import join

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch import nn
from tensorboardX import SummaryWriter
from loguru import logger
from tqdm import tqdm
from transformers import set_seed

from utils.args import get_parser, VersionConfig
from utils.optim import get_linear_schedule_with_warmup
from utils.utils import strftime, CountSmooth
from model.models import BertNER
from reader.nerReader import NERSet
from utils import ner


args = get_parser()
VERSION_CONFIG = VersionConfig(
    max_seq_length=args.max_seq_length
)
GPU_IDS = [0]
OUTPUT_DIR = join(args.output_dir, strftime())

if args.no_cuda or not torch.cuda.is_available():
    DEVICE = torch.device('cpu')
    logger.info('use cpu!')
    USE_CUDA = False
else:
    USE_CUDA = True
    DEVICE = torch.device('cuda', GPU_IDS[0])
    torch.cuda.set_device(DEVICE)
    logger.info('use gpu!')


def main(mode='dev'):
    assert mode in ['dev', 'test1', 'test2']

    # testset = NERSet(args, mode, False)
    # testloader = DataLoader(testset, batch_size=args.batch_size, num_workers=args.num_workers,
    #                          shuffle=True, collate_fn=NERSet.collate)

    model = torch.load(join(args.model_dir, 'model.pth'))

    # TODO 调试使用
    from train import evaluate
    print(evaluate(model))

if __name__ == '__main__':
    main('dev')