import torch
from torch import nn
from transformers import AutoModel, AutoModelWithLMHead
from reader.entityTypes import *
from allennlp.modules import ConditionalRandomField


class BertNER(nn.Module):
    def __init__(self, args, cfg):
        super(BertNER, self).__init__()

        self.encoder = AutoModel.from_pretrained(args.model_name_or_path, cache_dir=args.pretrained_cache_dir)
        self.hidden_size = self.encoder.config.hidden_size

        self.emission_ffn = nn.Linear(self.hidden_size, len(ID2LABEL))
        self.crossentropy = torch.nn.CrossEntropyLoss(ignore_index=LABEL2ID['[PAD]'])

    def forward(self, encoder_inputs: dict, label_ids: torch.Tensor):
        outputs = self.encoder(**encoder_inputs)
        # print(outputs[0].shape)
        encoded, _ = outputs
        
        emission = self.emission_ffn(encoded)
        emission = emission.permute(1,2,0)
        label_ids = label_ids.permute(1, 0)

        loss = self.crossentropy(emission, label_ids)
        return loss


    def predict(self, inputs: dict):
        if 'label_names' in inputs:
            inputs.pop('label_names')
        outputs = self.encoder(**inputs)
        encoded, _ = outputs
        emission = self.emission_ffn(encoded)
        pred_ids = torch.argmax(emission, -1)
        masks = inputs['attention_mask']

        # viterbi_decode = self.crf.viterbi_tags(emission, inputs['attention_mask'])
        # viterbi_path = [d[0] for d in viterbi_decode]
        tag_paths = []
        for i in range(emission.shape[0]):
            tag_paths.append([ID2LABEL[idx] for idx in pred_ids[i][:sum(masks[i])]])
        return tag_paths


