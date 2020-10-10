from typing import List, Dict, Tuple, Set

def decode_pred_seqs(batch_tag_paths: List[List[str]], sample_infos: List) -> Dict[str, Set[Tuple]]:
    """
    从预测的tag序列解码
    :param pred_tag_paths: keys are text_id where values were splitted from, shape: (batch, seq_lens)
    :return: 4-tuple of entites(tag_type, tag_start, tag_end, tag_text)
    """
    assert len(batch_tag_paths) == len(sample_infos)
    entitys = {}
    for tags, info in zip(batch_tag_paths, sample_infos):
        text = info['text']
        locs = info['token_loc_ids']  # 映射到原句的位置

        f_id, tag_type, tag_start, tag_end, tag_text = info['fid'], None, None, None, None
        tmp_start, tmp_end = None, None

        if not f_id in entitys:
            entitys[f_id] = set()

        i = 0
        while i < len(tags):
            if tags[i].startswith('B-'):
                tag_type = tags[i][2:]
                tag_start = locs[i]
                tmp_start = i

                i += 1
                while i < len(tags) and tags[i] == f'I-{tag_type}':
                    tag_end = locs[i]
                    tmp_end = i
                    i += 1

                tag_text = text[tmp_start : tmp_end]
                entitys[f_id].add((tag_type, tag_start, tag_end, tag_text))
            else:
                i += 1
    return entitys


