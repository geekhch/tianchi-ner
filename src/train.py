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

from reader.entityTypes import LABEL2ID
from utils.args import get_parser, VersionConfig
from utils.optim import get_linear_schedule_with_warmup
from utils.utils import strftime, CountSmooth
from model.models import BertNER
from reader.nerReader import NERSet, KFoldsWrapper, ThreadWrapper
from utils import ner

args = get_parser()
VERSION_CONFIG = VersionConfig(
    max_seq_length=args.max_seq_length,
    encoder_model=args.model_name_or_path,
    use_crf=args.use_crf,
    k_folds=args.k_folds
)
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


def evaluate(model, devloader, debug=False):
    model.eval()
    tp, fp, fn = 0, 0, 0
    with tqdm(total=len(devloader)) as t:
        t.set_description(f'EVAL')
        for model_inputs, sample_infos in devloader:
            label_names = model_inputs.pop('label_names')
            if USE_CUDA:
                for k, v in model_inputs.items():
                    if isinstance(v, torch.Tensor):
                        model_inputs[k] = v.cuda(DEVICE)
            pred_tag_seq = model.predict(model_inputs)
            batch_decode_labels = ner.decode_pred_seqs(pred_tag_seq, sample_infos)
            for decode_labels, sample_info in zip(batch_decode_labels, sample_infos):
                for lb in sample_info['gold']:
                    if not lb in decode_labels:
                        if debug:
                            print('FN:', lb)
                        fn += 1
                    else:
                        tp += 1
                for lb in decode_labels:
                    if not lb in sample_info['gold']:
                        if debug:
                            print('FP:', lb)
                        fp += 1
                if debug:
                    print(pred_tag_seq[0])
                    print(sample_info['fid'], sample_info['gold'], sample_info['text'], end='\n\n')
        t.update(len(devloader))
        t.set_postfix(tp=tp, fp=fp, fn=fn)
    model.train()

    percision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * (percision * recall) / (percision + recall + 1e-7)
    return percision, recall, f1


def main():
    writer = SummaryWriter(join(args.log_dir, strftime()))
    logger.info(f"output dir is: {OUTPUT_DIR}")

    set_seed(args.random_seed)
    model = BertNER(args, VERSION_CONFIG)

    if args.k_folds is None:
        trainset = NERSet(args, VERSION_CONFIG, 'train', True)
        devset = NERSet(args, VERSION_CONFIG, 'dev', True)
    else:
        logger.info(f"use k-folds: {args.k_folds}...")
        dev_fold_id, num_folds = args.k_folds.split('/')
        dev_fold_id, num_folds = int(dev_fold_id), int(num_folds)
        kfTool = KFoldsWrapper(args, VERSION_CONFIG)
        trainset, devset = kfTool.split_train_dev(num_folds, dev_fold_id)

    devloader = DataLoader(devset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False,
                           collate_fn=NERSet.collate)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, num_workers=args.num_workers,
                             shuffle=True, collate_fn=NERSet.collate)

    if USE_CUDA:
        model = model.cuda(DEVICE)
    optimizer = Adam([{'params': model.encoder.parameters()},
                      {'params': model.emission_ffn.parameters()},
                      {'params': model.crf.parameters(), "lr": 1e-2}], lr=args.learning_rate)

    scheduler = get_linear_schedule_with_warmup(optimizer, args.warmup_steps, args.max_steps)

    global_step = 0
    loss_ = CountSmooth(100)

    for epoch in range(args.max_epoches):
        trainWrapper = ThreadWrapper(trainloader, USE_CUDA, DEVICE)
        with tqdm(total=len(trainloader)) as t:
            t.set_description(f'Epoch {epoch}')
            model.train()
            for model_inputs, label_ids, sample_infos in trainWrapper.batch_generator():
                # label_names = model_inputs.pop('label_names')
                # label_ids = torch.tensor([[LABEL2ID[l_name] for l_name in line] for line in label_names]).cuda(DEVICE)
                # if USE_CUDA:
                #     for k, v in model_inputs.items():
                #         if isinstance(v, torch.Tensor):
                #             model_inputs[k] = v.cuda(DEVICE)

                global_step += 1
                loss = model(model_inputs, label_ids)
                loss_.add(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                writer.add_scalar('crf-loss', loss.item(), global_step=global_step)
                t.set_postfix(loss=loss_.get())
                t.update(1)

            # eval and save model every epoch
            p, r, f1 = evaluate(model, devloader)
            logger.info(f"after {global_step} steps,  percision={p}, recall={r}, f1={f1}\n")
            
            save_dir = join(OUTPUT_DIR, f'step_{global_step}')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            with open(join(save_dir, 'evaluate.txt'), 'w') as f:
                f.write(f'precision={p}, recall={r}, f1={f1} dev_size={len(devset)}\n')
                f.write(f'batch_size={args.batch_size}, epoch={epoch}, k_folds={args.k_folds}')
            torch.save(model, join(save_dir, 'model.pth'))
            VERSION_CONFIG.dump(save_dir)
                    


if __name__ == '__main__':
    main()
