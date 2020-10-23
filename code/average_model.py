import os, re, torch
from typing import List
from torch.nn import Module
from os.path import join

from utils.args import VersionConfig
from utils.utils import clear_dir

def merging(model_list: List[Module], w_list: List[float], cfg_list: List[VersionConfig]):
    D = sum(w_list)
    w_list = [w/D for w in w_list]
    assert len(model_list) == len(w_list)
    for M, w in zip(model_list, w_list):
        for param in M.parameters():
            param.data *= w

    i_M = model_list[0]
    i_CFG = cfg_list[0]
    for o_CFG, o_M in zip(cfg_list[1:], model_list[1:]):
        assert o_CFG.encoder_model == i_CFG.encoder_model
        assert o_CFG.max_seq_length == i_CFG.max_seq_length
        assert o_CFG.use_crf == i_CFG.use_crf
        for p1, p2 in zip(i_M.parameters(), o_M.parameters()):
            p1.data += p2.data

    return i_M, i_CFG

def read_save(save_dir):
    '''读取一个文件夹下不同step的模型'''
    model_list = []
    w_list = []
    cfg_list = []
    for root, dirs, files in os.walk(save_dir):
        if 'merge_model' in root:
            continue
        for fn in files:
            if fn != 'model.pth':
                continue
            eval_info = open(join(root, 'evaluate.txt')).read()
            f1 = float(re.findall(string=eval_info, pattern=r'f1=([0-9]\.[0-9]*)')[0])
            if f1 < 0.65:
                continue  # 放弃垃圾模型权重
            model = torch.load(join(root, 'model.pth'), map_location=torch.device('cpu'))
            model_list.append(model)
            w_list.append(f1)
            cfg = VersionConfig()
            cfg.load(root)
            cfg_list.append(cfg)
    assert len(model_list) == len(w_list) > 0

    i_M, i_CFG = merging(model_list, w_list, cfg_list)

    merge_out_dir = join(save_dir, 'merge_model')
    if os.path.exists(merge_out_dir):
        clear_dir(merge_out_dir)
    else:
        os.makedirs(merge_out_dir)
    torch.save(i_M, join(merge_out_dir, 'model.pth'))
    i_CFG.dump(merge_out_dir)


if __name__ == '__main__':
    read_save('output/10-14_17-07-35')
