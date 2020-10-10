import torch
from torch import nn
from transformers import AutoModel
from reader.entityTypes import *
from allennlp.modules import ConditionalRandomField


class BertNER(nn.Module):
    def __init__(self, args, USE_CUDA):
        super(BertNER, self).__init__()

        self.USE_CUDA=USE_CUDA

        self.encoder = AutoModel.from_pretrained(args.model_name_or_path, cache_dir=args.pretrained_cache_dir)
        self.hidden_size = self.encoder.config.hidden_size

        self.emission_ffn = nn.Linear(self.hidden_size, len(ID2LABEL))
        self.crf = ConditionalRandomField(len(ID2LABEL), include_start_end_transitions=False)


    def forward(self, inputs: dict):
        label_names = inputs.pop('label_names', None)
        encoded, _ = self.encoder(**inputs)

        emission = self.emission_ffn(encoded)
        batch_size = emission.shape[0]

        if label_names is None:
            viterbi_path = self.crf.viterbi_tags(emission, inputs['attention_mask'])
            return viterbi_path
        else:
            label_ids = torch.tensor([[LABEL2ID[l_name] for l_name in line] for line in label_names])
            if self.USE_CUDA:
                label_ids = label_ids.cuda()
            log_like_hood = self.crf(emission, label_ids, inputs['attention_mask'])
            log_like_hood /= batch_size
            return -log_like_hood

    def eval_segment_acc(self, emission, mask):
        viterbi_path = self.crf.viterbi_tags(emission, inputs['attention_mask'])
        
