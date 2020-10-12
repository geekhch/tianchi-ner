import torch
from torch import nn
from transformers import AutoModel, AutoModelWithLMHead
from reader.entityTypes import *
from allennlp.modules import ConditionalRandomField


class BertNER(nn.Module):
    def __init__(self, args, version_cfg, USE_CUDA, DEVICE=None):
        super(BertNER, self).__init__()

        self.USE_CUDA=USE_CUDA
        self.DEVICE = DEVICE if DEVICE else torch.device('cuda', 0)
        self.encoder = AutoModel.from_pretrained(args.model_name_or_path, cache_dir=args.pretrained_cache_dir)
        self.hidden_size = self.encoder.config.hidden_size

        self.emission_ffn = nn.Linear(self.hidden_size, len(ID2LABEL))
        self.crf = ConditionalRandomField(len(ID2LABEL), include_start_end_transitions=False)
        
        # 将转移矩阵参数冻结，相当于不使用CRF
        if not args.use_crf:
            self.crf.transitions.requires_grad=False
            torch.nn.init.ones_(self.crf.transitions)


    def forward(self, inputs: dict):
        label_names = inputs.pop('label_names', None)
        outputs = self.encoder(**inputs)
        # print(outputs[0].shape)
        encoded, _ = outputs

        emission = self.emission_ffn(encoded)
        batch_size = emission.shape[0]

        label_ids = [[LABEL2ID[l_name] for l_name in line] for line in label_names]
        tag_acc = self.eval_segment_acc(emission, inputs['attention_mask'], label_ids)

        label_ids = torch.tensor(label_ids)
        if self.USE_CUDA:
            label_ids = label_ids.cuda(self.DEVICE)
        log_like_hood = self.crf(emission, label_ids, inputs['attention_mask'])
        log_like_hood /= batch_size
        return -log_like_hood, tag_acc

    def eval_segment_acc(self, emission, mask, label_ids):
        viterbi_decode = self.crf.viterbi_tags(emission, mask)
        viterbi_paths = [d[0] for d in viterbi_decode]
        tot, tp = 0., 0.
        for i, path in enumerate(viterbi_paths):
            for j, tag_id in enumerate(path):
                tot += 1
                tp += tag_id == label_ids[i][j]
        return tp / tot

    def predict(self, inputs: dict):
        if 'label_names' in inputs:
            inputs.pop('label_names')
        outputs = self.encoder(**inputs)
        encoded, _ = outputs
        emission = self.emission_ffn(encoded)
        viterbi_decode = self.crf.viterbi_tags(emission, inputs['attention_mask'])
        viterbi_path = [d[0] for d in viterbi_decode]
        tag_paths = []
        for i in range(emission.shape[0]):
            tag_paths.append([ID2LABEL[idx] for idx in viterbi_path[i]])
        return tag_paths