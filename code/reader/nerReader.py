import time

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AlbertTokenizer, BertTokenizer
import os, re
from utils.utils import *
from loguru import logger
from typing import List, Tuple


class NERSet(Dataset):
    ''' 天池中医药NER任务：
    https://tianchi.aliyun.com/competition/entrance/531824/information
    '''

    @print_execute_time
    def __init__(self, args, version_cfg, MODE, FOR_TRAIN, file_paths: List[Tuple]=None):
        '''
        :param args:
        :param version_cfg:
        :param MODE: str of {train, dev ,test...}
        :param FOR_TRAIN:  Bool
        :param file_paths:  list of Tuples(dir, fid)
        '''
        assert MODE in ['train', 'dev', 'test1', 'test2']
        logger.info(f'begin reading {MODE}')

        self.FOR_TRAIN = FOR_TRAIN
        self.MODE = MODE
        self.args = args
        self.max_length = args.max_seq_length
        if version_cfg.encoder_model == 'hfl/chinese-bert-wwm-ext':
            self.tokenizer = AutoTokenizer.from_pretrained(version_cfg.encoder_model,
                                                           cache_dir=args.pretrained_cache_dir)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path,
                                                           cache_dir=args.pretrained_cache_dir)
        self.tokenizer.strip_accents = False

        # init file list
        if file_paths is None:
            filedir = os.path.join(self.args.data_dir, self.MODE)
            fids = set([fn.split('.')[0] for fn in os.listdir(filedir)])
            self.file_paths = [(filedir, fid) for fid in fids]
        else:
            self.file_paths = file_paths

        self.samples = self._load_train_data() if FOR_TRAIN else self._load_eval_data()

    def _load_train_data(self):
        samples = []
        for filedir, fid in self.file_paths:
            fp1 = os.path.join(filedir, f'{fid}.txt')
            fp2 = os.path.join(filedir, f'{fid}.ann')

            text = open(fp1, encoding='utf8').read()
            labels = ['O'] * len(text)
            all_tags = {}

            ann = map(str.split, open(fp2, encoding='utf8').readlines())
            for _, tag_type, tag_start, tag_end, tag_text in ann:
                all_tags[int(tag_start)] = (tag_type, tag_start, tag_end, tag_text)
                tag_start, tag_end = int(tag_start), int(tag_end)
                # 经检验，无重叠实体
                labels[tag_start] = f'B-{tag_type}'
                for i in range(tag_start + 1, tag_end):
                    labels[i] = f'I-{tag_type}'

            short_text = []
            short_text_labels = []
            short_text_loc_ids = []
            gold_tags = set()
            for i, token in enumerate(text):
                if token == '。' and len(short_text) > 0:
                    sample = {
                        'fid': fid,
                        'text': ' '.join(short_text),
                        'raw_doc': text,
                        'token_loc_ids': short_text_loc_ids.copy(),
                        'gold': gold_tags.copy()
                    }
                    _tmp_labels = short_text_labels.copy()
                    samples.append((sample, _tmp_labels))
                    short_text.clear()
                    short_text_labels.clear()
                    short_text_loc_ids.clear()
                    gold_tags.clear()
                elif not re.match('\s', token):  # 非空白符
                    short_text.append(token)
                    short_text_loc_ids.append(i)
                    short_text_labels.append(labels[i])
                    if i in all_tags:
                        gold_tags.add(all_tags[i])

            if len(short_text) > 1:
                sample = {
                    'fid': fid,
                    'text': ' '.join(short_text),
                    'raw_doc': text,
                    'token_loc_ids': short_text_loc_ids.copy(),
                    'gold': gold_tags.copy()
                }
                _tmp_labels = short_text_labels.copy()
                samples.append((sample, _tmp_labels))
        return samples

    def _load_eval_data(self):
        samples = []
        for filedir, fid in self.file_paths:
            fp1 = os.path.join(filedir, f'{fid}.txt')
            text = open(fp1, encoding='utf8').read()

            short_text = []
            short_text_loc_ids = []
            for i, token in enumerate(text):
                if token == '。' and len(short_text) > 1:
                    sample = {
                        'fid': fid,
                        'text': ' '.join(short_text),
                        'raw_doc': text,
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
                    'raw_doc': text,
                    'token_loc_ids': short_text_loc_ids.copy()
                }
                samples.append(sample)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        if self.FOR_TRAIN:
            sample, labels = self.samples[i]
            labels = labels[:self.max_length - 2]
            label_length = len(labels) + 2
        else:
            sample = self.samples[i].copy()

        sample['token_loc_ids'] = [sample['token_loc_ids'][0]] + sample['token_loc_ids'] + [
            sample['token_loc_ids'][-1] + 1]  # for cls and seq
        sample['token_loc_ids'] = sample['token_loc_ids'][:self.max_length]

        sample_encoding = self.tokenizer.encode_plus(sample['text'], padding='max_length',
                                                     max_length=self.max_length, truncation=True,
                                                     return_attention_mask=True)

        assert len(sample['token_loc_ids']) == sum(sample_encoding['attention_mask'])

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


class KFoldsWrapper:
    def __init__(self, args, cfg):
        self.seed = args.random_seed
        self.args = args
        self.cfg = cfg
        self._load_data_list()

    def _load_data_list(self):
        file_set = set()

        filedir = os.path.join(self.args.data_dir, 'train')
        assert os.path.exists(filedir)
        for fn in os.listdir(filedir):
            fn_ = fn.split('.')[0]
            file_set.add((filedir, fn_))

        filedir = os.path.join(self.args.data_dir, 'dev')
        assert os.path.exists(filedir)
        for fn in os.listdir(filedir):
            fn_ = fn.split('.')[0]
            file_set.add((filedir, fn_))

        file_list = list(file_set)
        file_list.sort(key=lambda x: int(x[1]))
        import random
        random.seed(self.seed)
        random.shuffle(file_list)

        self._file_list = file_list

    def split_train_dev(self, num_folds, dev_fold_id):
        assert 0 <= dev_fold_id < num_folds
        fold_size = len(self._file_list) // num_folds

        dev_start = fold_size * dev_fold_id
        if dev_fold_id == num_folds - 1:
            dev_end = len(self._file_list)
        else:
            dev_end = dev_start + fold_size

        train_list = self._file_list[:dev_start] + self._file_list[dev_end:]
        dev_list = self._file_list[dev_start:dev_end]
        trainset = NERSet(self.args, self.cfg, 'train', True, train_list)
        devset = NERSet(self.args, self.cfg, 'dev', True, dev_list)

        return trainset, devset

import threading, queue
class ThreadWrapper:
    '''load to gpu in muti-threads: 异步将cpu数据加载到GPU'''
    def __init__(self, dataloader, USE_CUDA, DEVICE):
        self.USE_CUDA = USE_CUDA
        self.DEVICE = DEVICE
        self.q = queue.Queue(3)
        self.dataloader = dataloader
        T = threading.Thread(target=self._gen)
        T.start()
        self.batch_id = 0


    def _gen(self):
        for model_inputs, sample_infos in self.dataloader:
            label_names = model_inputs.pop('label_names')
            from reader.entityTypes import LABEL2ID
            label_ids = torch.tensor([[LABEL2ID[l_name] for l_name in line] for line in label_names]).cuda(self.DEVICE)
            if self.USE_CUDA:
                for k, v in model_inputs.items():
                    if isinstance(v, torch.Tensor):
                        model_inputs[k] = v.cuda(self.DEVICE)
            batch = (model_inputs, label_ids, sample_infos)
            self.q.put(batch)

    def batch_generator(self):
        while self.batch_id < len(self.dataloader):
            ret = self.q.get()
            self.batch_id += 1
            yield ret




if __name__ == '__main__':
    from utils.args import get_parser

    args = get_parser()


    class Ver:
        encoder_model = 'hfl/chinese-bert-wwm-ext'

    dataset = NERSet(args, Ver, 'dev', True)
    dataloader = DataLoader(dataset, batch_size=3, collate_fn=NERSet.collate)
    train_wrapper = ThreadWrapper(dataloader, True, torch.device('cuda',0))
    total = 0
    for batch in train_wrapper.batch_generator():
        total += len(batch[2])
        print(batch)
    print(total, len(dataset))
