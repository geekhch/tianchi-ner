import torch
from torch import nn
from allennlp.modules import ConditionalRandomField
from transformers import AutoModel, AutoModelWithLMHead
from reader.entityTypes import *


class BertNER(nn.Module):
    def __init__(self, args, cfg):
        super(BertNER, self).__init__()

        self.encoder = AutoModel.from_pretrained(args.model_name_or_path, cache_dir=args.pretrained_cache_dir)
        self.embeddings = self.encoder.get_input_embeddings()
        self.hidden_size = self.encoder.config.hidden_size

        self.emission_ffn = nn.Linear(self.hidden_size, len(ID2LABEL))
        self.crf = ConditionalRandomField(len(ID2LABEL), include_start_end_transitions=False)
        
        # 将转移矩阵参数冻结，相当于不使用CRF
        if not cfg.use_crf:
            self.crf.transitions.requires_grad=False
            torch.nn.init.zeros_(self.crf.transitions)

    def forward(self, encoder_inputs: dict, label_ids: torch.Tensor):
        input_ids_1 = encoder_inputs['input_ids_1']
        input_ids_2 = encoder_inputs['input_ids_2']
        
        loss_mask = encoder_inputs['loss_mask']
        attention_mask = encoder_inputs['attention_mask']
        token_type_ids = encoder_inputs['token_type_ids']
        position_ids = encoder_inputs['position_ids']
        
        embed_1 = self.embeddings(input_ids_1)
        embed_2 = self.embeddings(input_ids_2)
        embed = (embed_1 + embed_2) / 2

        
        
        outputs = self.encoder(inputs_embeds=embed, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids)
        # print(outputs[0].shape)
        encoded, _ = outputs

        emission = self.emission_ffn(encoded)
        batch_size = emission.shape[0]

        log_like_hood = self.crf(emission, label_ids, loss_mask)
        log_like_hood /= batch_size
        # return -log_like_hood, tag_acc
        return -log_like_hood

    def eval_segment_acc(self, emission, mask, label_ids):
        viterbi_decode = self.crf.viterbi_tags(emission, mask)
        viterbi_paths = [d[0] for d in viterbi_decode]
        tot, tp = 0., 0.
        for i, path in enumerate(viterbi_paths):
            for j, tag_id in enumerate(path):
                tot += 1
                tp += tag_id == label_ids[i][j]
        return tp / tot

    def predict(self, encoder_inputs: dict):
        if 'label_names' in encoder_inputs:
            encoder_inputs.pop('label_names')
        
        input_ids_1 = encoder_inputs['input_ids_1']
        input_ids_2 = encoder_inputs['input_ids_2']
        
        loss_mask = encoder_inputs['loss_mask']
        attention_mask = encoder_inputs['attention_mask']
        token_type_ids = encoder_inputs['token_type_ids']
        position_ids = encoder_inputs['position_ids']
        
        embed_1 = self.embeddings(input_ids_1)
        embed_2 = self.embeddings(input_ids_2)
        embed = (embed_1 + embed_2) / 2


        outputs = self.encoder(inputs_embeds=embed, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids)
        encoded, _ = outputs
        emission = self.emission_ffn(encoded)
        viterbi_decode = self.crf.viterbi_tags(emission, loss_mask)
        viterbi_path = [d[0] for d in viterbi_decode]
        tag_paths = []
        for i in range(emission.shape[0]):
            tag_paths.append([ID2LABEL[idx] for idx in viterbi_path[i]])
        return tag_paths

    def predict_k(self, inputs: dict):
        if 'label_names' in inputs:
            inputs.pop('label_names')
        outputs = self.encoder(**inputs)
        encoded, _ = outputs
        emission = self.emission_ffn(encoded)

        viterbi_decode_k = self.crf.viterbi_tags(emission, inputs['attention_mask'], top_k=3)
        viterbi_path_k = [[d[0] for d in viterbi_decode] for viterbi_decode in viterbi_decode_k]
        tag_paths_k = []
        for i in range(emission.shape[0]): # sample
            pk = []
            for j in range(3):
                # top 3, 如果后面的预测分数小于前面的一半就扔掉
                if j > 0 and viterbi_decode_k[i][j][1] < viterbi_decode_k[i][0][1]/2:
                    continue
                pk.append([ID2LABEL[idx] for idx in viterbi_path_k[i][j]])
            tag_paths_k.append(pk)

        return tag_paths_k
