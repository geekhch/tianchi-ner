# export ENCODER=voidful/albert_chinese_small

# python src/train.py \
# --model_name_or_path $ENCODER \
# --pretrained_cache_dir ./pretrained \
# --batch_size 8 \
# --lr 5e-5 \
# --max_epoches 10 \
# --max_steps 20000 \
# --warmup_steps 100 \
# --num_workers 2 \
# --save_steps 200 \
# --gpu_id 1 \
# --use_crf

# 不使用crf
# export ENCODER=hfl/chinese-bert-wwm-ext
export ENCODER=hfl/chinese-roberta-wwm-ext

python src/train.py \
--model_name_or_path $ENCODER \
--pretrained_cache_dir ./pretrained \
--batch_size 8 \
--lr 5e-5 \
--max_epoches 5 \
--max_steps 20000 \
--warmup_steps 100 \
--num_workers 4 \
--gpu_id 1 \
--use_crf \
--k_folds 4/10

