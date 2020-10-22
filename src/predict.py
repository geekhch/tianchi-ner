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
from utils.utils import strftime, CountSmooth, clear_dir
from model.models import BertNER
from reader.nerReader import NERSet
from utils import ner


args = get_parser()
VERSION_CONFIG = VersionConfig()
VERSION_CONFIG.load(args.model_dir)
GPU_IDS = [args.gpu_id]
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
    logger.info(f"load model from {args.model_dir}")
    model = torch.load(join(args.model_dir, 'model.pth'), map_location=DEVICE)
    model.eval()

    # TODO 调试使用
    # from train import evaluate
    # devset = NERSet(args, VERSION_CONFIG, 'dev', True)
    # devloader = DataLoader(devset, batch_size=args.batch_size, collate_fn=NERSet.collate)
    # print(evaluate(model, devloader, debug=True))

    def predict():
        testset = NERSet(args, VERSION_CONFIG, mode, False)
        testloader = DataLoader(testset, batch_size=args.batch_size,  collate_fn=NERSet.collate)
        file2entities = {}
        with tqdm(total=len(testloader), ncols=50) as t:
            t.set_description(f'EVAL')
            for model_inputs, sample_infos in testloader:
                if USE_CUDA:
                    for k, v in model_inputs.items():
                        if isinstance(v, torch.Tensor):
                            model_inputs[k] = v.cuda(DEVICE)
                if args.topk:
                    pred_tag_seq = model.predict_k(model_inputs)
                    batch_decode_labels = ner.decode_topk_pred_seqs(pred_tag_seq, sample_infos)
                else:
                    pred_tag_seq = model.predict(model_inputs)
                    batch_decode_labels = ner.decode_pred_seqs(pred_tag_seq, sample_infos)

                for decode_labels, sample_info in zip(batch_decode_labels, sample_infos):
                    fid = sample_info['fid']
                    if fid not in file2entities:
                        file2entities[fid] = set()
                    for lb in decode_labels:
                        file2entities[fid].add(lb)
                t.update(1)

        save_dir = join(args.model_dir, f'predict_{mode}')
        if os.path.exists(save_dir):
            clear_dir(save_dir)
        else:
            os.mkdir(save_dir)
        for fid, E in file2entities.items():
            with open(join(save_dir, f'{fid}.ann'), 'w', encoding='utf8') as f:
                firstline = True
                for i, e in enumerate(E):
                    if not firstline:
                        f.write('\n')
                    f.write(f'T{i+1}\t{e[0]} {e[1]} {e[2]}\t{e[3]}')
                    firstline=False

    predict()


if __name__ == '__main__':
    main('dev')