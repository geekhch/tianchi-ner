import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AlbertTokenizer, BertTokenizer
import os, re
from utils.utils import *
from loguru import logger

class NERSet(Dataset):
    ''' 天池中医药NER任务：
    https://tianchi.aliyun.com/competition/entrance/531824/information
    '''

    @print_execute_time
    def __init__(self, args, MODE: str, FOR_TRAIN: bool, tokenizer=None):
        assert MODE in ['train', 'dev', 'test1', 'test2']

        self.FOR_TRAIN = FOR_TRAIN
        self.MODE = MODE
        self.args = args
        self.max_length = args.max_seq_length

        self.tokenizer = tokenizer if tokenizer else BertTokenizer.from_pretrained(
            args.model_name_or_path, cache_dir=args.pretrained_cache_dir)
        self.tokenizer.strip_accents = False

        logger.info(f'begin reading {MODE}')
        self.samples = self._load_train_data() if FOR_TRAIN else self._load_eval_data()

    def _load_train_data(self):
        filedir = os.path.join(self.args.data_dir, self.MODE)
        fids = set([fn.split('.')[0] for fn in os.listdir(filedir)])

        samples = []
        for fid in fids:
            fp1 = os.path.join(filedir, f'{fid}.txt')
            fp2 = os.path.join(filedir, f'{fid}.ann')

            text = open(fp1, encoding='utf8').read()
            labels = ['O'] * len(text)
            gold_tags = set()

            ann = map(str.split, open(fp2, encoding='utf8').readlines())
            for _, tag_type, tag_start, tag_end, tag_text in ann:
                gold_tags.add((fid, tag_type, tag_start, tag_end, tag_text))
                tag_start, tag_end = int(tag_start), int(tag_end)
                # 经检验，无重叠实体
                labels[tag_start] = f'B-{tag_type}'
                for i in range(tag_start + 1, tag_end):
                    labels[i] = f'I-{tag_type}'

            short_text = []
            short_text_labels = []
            short_text_loc_ids = []
            for i, token in enumerate(text):
                if token == '。' and len(short_text) > 0:
                    sample = {
                        'fid': fid,
                        'text': ' '.join(short_text),
                        'token_loc_ids': short_text_loc_ids.copy(),
                        'gold': gold_tags
                    }
                    _tmp_labels = short_text_labels.copy()
                    samples.append((sample, _tmp_labels))
                    short_text.clear()
                    short_text_labels.clear()
                    short_text_loc_ids.clear()
                elif not re.match('\s', token):  # 非空白符
                    short_text.append(token)
                    short_text_loc_ids.append(i)
                    short_text_labels.append(labels[i])

            if len(short_text) > 1:
                sample = {
                    'fid': fid,
                    'text': ' '.join(short_text),
                    'token_loc_ids': short_text_loc_ids.copy(),
                    'gold': gold_tags
                }
                _tmp_labels = short_text_labels.copy()
                samples.append((sample, _tmp_labels))
        return samples

    def _load_eval_data(self):
        filedir = os.path.join(self.args.data_dir, self.MODE)
        fids = set([fn.split('.')[0] for fn in os.listdir(filedir)])

        samples = []
        for fid in fids:
            fp1 = os.path.join(filedir, f'{fid}.txt')
            text = open(fp1, encoding='utf8').read()

            short_text = []
            short_text_loc_ids = []
            for i, token in enumerate(text):
                if token == '。' and len(short_text) > 1:
                    sample = {
                        'fid': fid,
                        'text': ' '.join(short_text),
                        'token_loc_ids': short_text_loc_ids.copy()
                    }
                    short_text.clear()
                    short_text_loc_ids.clear()
                    samples.append(sample)
                elif not re.match('\s', token):  # 非空白符
                    short_text.append(token)
                    short_text_loc_ids.append(i)

            if len(short_text) > 1:
                sample = {
                    'fid': fid,
                    'text': ' '.join(short_text),
                    'token_loc_ids': short_text_loc_ids.copy()
                }
                samples.append(sample)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        if self.FOR_TRAIN:
            sample, labels = self.samples[i]
            labels = labels[:self.max_length-2]
            label_length = len(labels) + 2
        else:
            sample = self.samples[i]

        sample_encoding = self.tokenizer.encode_plus(sample['text'], padding='max_length',
                                                     max_length=self.max_length, truncation=True,
                                                     return_attention_mask=True)

        if self.FOR_TRAIN:
            sample_encoding['label_names'] = ['[CLS]'] + labels + ['[SEP]'] \
                                             + ['[PAD]'] * (self.max_length - label_length)
        return sample_encoding, sample

    @staticmethod
    def collate(batch):
        _elem = batch[0][0]
        sample_infos = [s[1] for s in batch]
        model_inputs = {}
        for k in _elem.keys():
            for s in batch:
                if not k in model_inputs:
                    model_inputs[k] = []
                model_inputs[k].append(s[0][k])

            # convert to tensor
            if isinstance(model_inputs[k][0], list):
                if isinstance(model_inputs[k][0][0], int):
                    model_inputs[k] = torch.tensor(model_inputs[k])
                elif isinstance(model_inputs[k][0][0], float):
                    model_inputs[k] = torch.tensor(model_inputs[k], dtype=torch.float64)

        return model_inputs, sample_infos

if __name__ == '__main__':
    from utils.args import get_parser

    args = get_parser()

    dataset = NERSet(args, 'dev', True)
    dataloader = DataLoader(dataset, batch_size=3, collate_fn=NERSet.collate)
    for batch in dataloader:
        print(batch)
