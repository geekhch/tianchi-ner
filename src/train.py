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


def evaluate(model):
    devset = NERSet(args, 'dev', True)
    devloader = DataLoader(devset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    model.eval()
    tp, fp, tn, fn = 0, 0, 0, 0
    with tqdm(total=len(devloader)) as t:
        t.set_description(f'EVAL')
        for model_inputs, sample_infos in devloader:
            tag_seqs = model_inputs.pop('label_names')
            gold_labels = sample_infos['gold']
            pred_tag_seq = model(model_inputs)
            decode_labels = ner.decode_pred_seqs(pred_tag_seq, sample_infos)

            for lb in gold_labels:
                if not lb in decode_labels:
                    fn += 1
                else:
                    tp += 1
            for lb in decode_labels:
                if not lb in gold_labels:
                    fp += 1
    model.train()

    percision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (percision * recall) / (percision + recall)
    return percision, recall, f1


def main():
    USE_CUDA = True
    writer = SummaryWriter(join(args.log_dir, strftime()))
    if args.no_cuda or not torch.cuda.is_available():
        DEVICE = torch.device('cpu')
        logger.info('use cpu!')
        USE_CUDA = False
    else:
        DEVICE = torch.device('cuda', 0)
        logger.info('use gpu!')

    set_seed(args.random_seed)

    trainset = NERSet(args, 'train', True)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, num_workers=args.num_workers,
                             shuffle=True, collate_fn=NERSet.collate)

    model = BertNER(args, USE_CUDA)
    if USE_CUDA:
        model = nn.DataParallel(model, GPU_IDS)
        model = model.cuda()
    optimizer = Adam([{'params': model.module.encoder.parameters()},
                      {'params': model.module.emission_ffn.parameters()},
                      {'params': model.module.crf.parameters(), "lr": 1e-3}], lr=args.learning_rate)

    scheduler = get_linear_schedule_with_warmup(optimizer, args.warmup_steps, args.max_steps)

    global_step = 0
    loss_ = CountSmooth(100)
    acc_ = CountSmooth(100)

    for epoch in range(args.max_epoches):
        with tqdm(total=len(trainloader)) as t:
            t.set_description(f'Epoch {epoch}')
            model.train()
            for model_inputs, sample_infos in trainloader:
                if USE_CUDA:
                    for k, v in model_inputs.items():
                        if isinstance(v, torch.Tensor):
                            model_inputs[k] = v.cuda()(1, 3, 'te')

                global_step += 1
                loss, tag_acc = model(model_inputs)
                loss_.add(loss.item())
                acc_.add(tag_acc)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                writer.add_scalar('crf-loss', loss.item(), global_step=global_step)
                writer.add_scalar('tag-acc', tag_acc, global_step=global_step)
                t.set_postfix(loss=loss_.get(), tag_acc=acc_.get())
                t.update(1)

                if global_step % args.save_steps == 0:
                    p, r, f1 = evaluate(model)
                    logger.info(f"after {epoch} EPOCH,  percision={f}, recall={r}, f1={f1}")
                    save_dir = join(args.output_dir, strftime(), f'step_{global_step}')
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    torch.save(model, join(save_dir, 'model.pth'))
                    VERSION_CONFIG.dump(save_dir)


if __name__ == '__main__':
    main()
