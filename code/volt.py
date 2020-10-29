import os, pickle
import random

fid_entity_count = {}

predict_version = []
for root, dirs, files in os.walk('./user_data'):
    for d in dirs:
        if not d.startswith('predict_test1'):
            continue
        predict_version.append(os.path.join(root, d))
print(predict_version)

for folder in predict_version:
    for fid in os.listdir(folder):
        if not fid in fid_entity_count:
            fid_entity_count[fid] = {}
        with open(f'{folder}/{fid}', encoding='utf8') as f:
            for line in f.readlines():
                line = line.strip()
                tid, m, e_text = line.split('\t')
                e_type, e_start, e_end = m.split(' ')
                E = (e_type, e_start, e_end, e_text)
                fid_entity_count[fid][E] = fid_entity_count[fid].get(E, 0)+1

# zgy = pickle.load(open('./user_data/merge_5e-5.pkl','rb'))
# for int_fid, e2c in zgy.items():
#     fid = f'{int_fid}.ann'
#     for E, C in e2c.items():
#         E = (E[0], str(E[1]), str(E[2]), E[3])
#         if not E in fid_entity_count[fid]:
#             fid_entity_count[fid][E] = C
#         else:
#             fid_entity_count[fid][E] += C

def clear_dir(dir):
    files = [os.path.join(dir, f) for f in os.listdir(dir)]
    for f in files:
        os.remove(f)


MIN_COUNT = 7
print(f"条件：count大于等于{MIN_COUNT}")

if not os.path.exists('./user_data/result'):
    os.mkdir('./user_data/result')
else:
    clear_dir('./user_data/result')

# 防止复赛空文件
for i in range(1500, 1997):
    with open(f'./user_data/result/{i}.ann', 'w', encoding='utf8') as f:
        f.write('')

num_entity = 0
for fid, entity_count in fid_entity_count.items():
    TID = 1
    with open(f'./user_data/result/{fid}', 'w', encoding='utf8') as f:
        for E, count in entity_count.items():
            if count < MIN_COUNT:
                continue
            if TID > 1:
                f.write('\n')
            e_type, e_start, e_end, e_text = E
            f.write(f'T{TID}\t{e_type} {e_start} {e_end}\t{e_text}')
            TID += 1
            num_entity += 1

print(f"count >= {MIN_COUNT}时平均每个样本有{num_entity/len(fid_entity_count)}个样本")
