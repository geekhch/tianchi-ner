# export ENCODER=voidful/albert_chinese_small
export ENCODER=hfl/chinese-bert-wwm-ext

python src/train.py \
--model_name_or_path $ENCODER \
--pretrained_cache_dir ./pretrained \
--batch_size 8 \
--lr 5e-5 \
--max_epoches 60 \
--max_steps 20000 \
--warmup_steps 100 \
--num_workers 1 \
--save_steps 50 \
--gpu_id 1