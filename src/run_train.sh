python src/train.py \
--model_name_or_path voidful/albert_chinese_small \
--pretrained_cache_dir ./pretrained \
--batch_size 32 \
--lr 5e-4 \
--max_epoches 40 \
--max_steps 20000 \
--warmup_steps 100 \
--num_workers 3