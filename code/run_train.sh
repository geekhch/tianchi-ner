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
# export ENCODER=hfl/chinese-roberta-wwm-ext-large
export OUTPUT_DIR=./output
# # export ENCODER=allenyummy/chinese-bert-wwm-ehr-ner-sl

python src/train.py --model_name_or_path $ENCODER --output_dir $OUTPUT_DIR --pretrained_cache_dir ./pretrained --batch_size 8 --lr 5e-5 --max_epoches 7 --max_steps 20000 --warmup_steps 100 --num_workers 1 --gpu_id 1

# KFolds 规则
# python src/train.py --model_name_or_path $ENCODER --output_dir $OUTPUT_DIR --pretrained_cache_dir ./pretrained --batch_size 8 --lr 5e-5 --max_epoches 5 --max_steps 20000 --warmup_steps 100 --num_workers 2 --gpu_id 1 --k_folds 0/40

# python src/train.py --model_name_or_path $ENCODER --output_dir $OUTPUT_DIR --pretrained_cache_dir ./pretrained --batch_size 8 --lr 5e-5 --max_epoches 5 --max_steps 20000 --warmup_steps 100 --num_workers 2 --gpu_id 1 --k_folds 1/40

# python src/train.py --model_name_or_path $ENCODER --output_dir $OUTPUT_DIR --pretrained_cache_dir ./pretrained --batch_size 8 --lr 5e-5 --max_epoches 5 --max_steps 20000 --warmup_steps 100 --num_workers 2 --gpu_id 1 --k_folds 2/40

# python src/train.py --model_name_or_path $ENCODER --output_dir $OUTPUT_DIR --pretrained_cache_dir ./pretrained --batch_size 8 --lr 5e-5 --max_epoches 5 --max_steps 20000 --warmup_steps 100 --num_workers 2 --gpu_id 1 --k_folds 3/40

# python src/train.py --model_name_or_path $ENCODER --output_dir $OUTPUT_DIR --pretrained_cache_dir ./pretrained --batch_size 8 --lr 5e-5 --max_epoches 5 --max_steps 20000 --warmup_steps 100 --num_workers 2 --gpu_id 1 --k_folds 4/40

# python src/train.py --model_name_or_path $ENCODER --output_dir $OUTPUT_DIR --pretrained_cache_dir ./pretrained --batch_size 8 --lr 5e-5 --max_epoches 5 --max_steps 20000 --warmup_steps 100 --num_workers 2 --gpu_id 1 --k_folds 5/40

# python src/train.py --model_name_or_path $ENCODER --output_dir $OUTPUT_DIR --pretrained_cache_dir ./pretrained --batch_size 8 --lr 5e-5 --max_epoches 5 --max_steps 20000 --warmup_steps 100 --num_workers 2 --gpu_id 1 --k_folds 6/40

# python src/train.py --model_name_or_path $ENCODER --output_dir $OUTPUT_DIR --pretrained_cache_dir ./pretrained --batch_size 8 --lr 5e-5 --max_epoches 5 --max_steps 20000 --warmup_steps 100 --num_workers 2 --gpu_id 1 --k_folds 7/40

# python src/train.py --model_name_or_path $ENCODER --output_dir $OUTPUT_DIR --pretrained_cache_dir ./pretrained --batch_size 8 --lr 5e-5 --max_epoches 5 --max_steps 20000 --warmup_steps 100 --num_workers 2 --gpu_id 1 --k_folds 8/40

# python src/train.py --model_name_or_path $ENCODER --output_dir $OUTPUT_DIR --pretrained_cache_dir ./pretrained --batch_size 8 --lr 5e-5 --max_epoches 5 --max_steps 20000 --warmup_steps 100 --num_workers 2 --gpu_id 1 --k_folds 9/40

# python src/train.py --model_name_or_path $ENCODER --output_dir $OUTPUT_DIR --pretrained_cache_dir ./pretrained --batch_size 8 --lr 5e-5 --max_epoches 5 --max_steps 20000 --warmup_steps 100 --num_workers 2 --gpu_id 1 --k_folds 10/40

# python src/train.py --model_name_or_path $ENCODER --output_dir $OUTPUT_DIR --pretrained_cache_dir ./pretrained --batch_size 8 --lr 5e-5 --max_epoches 5 --max_steps 20000 --warmup_steps 100 --num_workers 2 --gpu_id 1 --k_folds 11/40

# python src/train.py --model_name_or_path $ENCODER --output_dir $OUTPUT_DIR --pretrained_cache_dir ./pretrained --batch_size 8 --lr 5e-5 --max_epoches 5 --max_steps 20000 --warmup_steps 100 --num_workers 2 --gpu_id 1 --k_folds 12/40

# python src/train.py --model_name_or_path $ENCODER --output_dir $OUTPUT_DIR --pretrained_cache_dir ./pretrained --batch_size 8 --lr 5e-5 --max_epoches 5 --max_steps 20000 --warmup_steps 100 --num_workers 2 --gpu_id 1 --k_folds 13/40

# python src/train.py --model_name_or_path $ENCODER --output_dir $OUTPUT_DIR --pretrained_cache_dir ./pretrained --batch_size 8 --lr 5e-5 --max_epoches 5 --max_steps 20000 --warmup_steps 100 --num_workers 2 --gpu_id 1 --k_folds 14/40

# python src/train.py --model_name_or_path $ENCODER --output_dir $OUTPUT_DIR --pretrained_cache_dir ./pretrained --batch_size 8 --lr 5e-5 --max_epoches 5 --max_steps 20000 --warmup_steps 100 --num_workers 2 --gpu_id 1 --k_folds 15/40

# python src/train.py --model_name_or_path $ENCODER --output_dir $OUTPUT_DIR --pretrained_cache_dir ./pretrained --batch_size 8 --lr 5e-5 --max_epoches 5 --max_steps 20000 --warmup_steps 100 --num_workers 2 --gpu_id 1 --k_folds 16/40

# python src/train.py --model_name_or_path $ENCODER --output_dir $OUTPUT_DIR --pretrained_cache_dir ./pretrained --batch_size 8 --lr 5e-5 --max_epoches 5 --max_steps 20000 --warmup_steps 100 --num_workers 2 --gpu_id 1 --k_folds 17/40

# python src/train.py --model_name_or_path $ENCODER --output_dir $OUTPUT_DIR --pretrained_cache_dir ./pretrained --batch_size 8 --lr 5e-5 --max_epoches 5 --max_steps 20000 --warmup_steps 100 --num_workers 2 --gpu_id 1 --k_folds 18/40

# python src/train.py --model_name_or_path $ENCODER --output_dir $OUTPUT_DIR --pretrained_cache_dir ./pretrained --batch_size 8 --lr 5e-5 --max_epoches 5 --max_steps 20000 --warmup_steps 100 --num_workers 2 --gpu_id 1 --k_folds 19/40

# python src/train.py --model_name_or_path $ENCODER --output_dir $OUTPUT_DIR --pretrained_cache_dir ./pretrained --batch_size 8 --lr 5e-5 --max_epoches 5 --max_steps 20000 --warmup_steps 100 --num_workers 2 --gpu_id 1 --k_folds 20/40

# python src/train.py --model_name_or_path $ENCODER --output_dir $OUTPUT_DIR --pretrained_cache_dir ./pretrained --batch_size 8 --lr 5e-5 --max_epoches 5 --max_steps 20000 --warmup_steps 100 --num_workers 2 --gpu_id 1 --k_folds 21/40

# python src/train.py --model_name_or_path $ENCODER --output_dir $OUTPUT_DIR --pretrained_cache_dir ./pretrained --batch_size 8 --lr 5e-5 --max_epoches 5 --max_steps 20000 --warmup_steps 100 --num_workers 2 --gpu_id 1 --k_folds 22/40

# python src/train.py --model_name_or_path $ENCODER --output_dir $OUTPUT_DIR --pretrained_cache_dir ./pretrained --batch_size 8 --lr 5e-5 --max_epoches 5 --max_steps 20000 --warmup_steps 100 --num_workers 2 --gpu_id 1 --k_folds 23/40

# python src/train.py --model_name_or_path $ENCODER --output_dir $OUTPUT_DIR --pretrained_cache_dir ./pretrained --batch_size 8 --lr 5e-5 --max_epoches 5 --max_steps 20000 --warmup_steps 100 --num_workers 2 --gpu_id 1 --k_folds 24/40

# python src/train.py --model_name_or_path $ENCODER --output_dir $OUTPUT_DIR --pretrained_cache_dir ./pretrained --batch_size 8 --lr 5e-5 --max_epoches 5 --max_steps 20000 --warmup_steps 100 --num_workers 2 --gpu_id 1 --k_folds 25/40

# python src/train.py --model_name_or_path $ENCODER --output_dir $OUTPUT_DIR --pretrained_cache_dir ./pretrained --batch_size 8 --lr 5e-5 --max_epoches 5 --max_steps 20000 --warmup_steps 100 --num_workers 2 --gpu_id 1 --k_folds 26/40

# python src/train.py --model_name_or_path $ENCODER --output_dir $OUTPUT_DIR --pretrained_cache_dir ./pretrained --batch_size 8 --lr 5e-5 --max_epoches 5 --max_steps 20000 --warmup_steps 100 --num_workers 2 --gpu_id 1 --k_folds 27/40

# python src/train.py --model_name_or_path $ENCODER --output_dir $OUTPUT_DIR --pretrained_cache_dir ./pretrained --batch_size 8 --lr 5e-5 --max_epoches 5 --max_steps 20000 --warmup_steps 100 --num_workers 2 --gpu_id 1 --k_folds 28/40

# python src/train.py --model_name_or_path $ENCODER --output_dir $OUTPUT_DIR --pretrained_cache_dir ./pretrained --batch_size 8 --lr 5e-5 --max_epoches 5 --max_steps 20000 --warmup_steps 100 --num_workers 2 --gpu_id 1 --k_folds 29/40

# python src/train.py --model_name_or_path $ENCODER --output_dir $OUTPUT_DIR --pretrained_cache_dir ./pretrained --batch_size 8 --lr 5e-5 --max_epoches 5 --max_steps 20000 --warmup_steps 100 --num_workers 2 --gpu_id 1 --k_folds 30/40

# python src/train.py --model_name_or_path $ENCODER --output_dir $OUTPUT_DIR --pretrained_cache_dir ./pretrained --batch_size 8 --lr 5e-5 --max_epoches 5 --max_steps 20000 --warmup_steps 100 --num_workers 2 --gpu_id 1 --k_folds 31/40

# python src/train.py --model_name_or_path $ENCODER --output_dir $OUTPUT_DIR --pretrained_cache_dir ./pretrained --batch_size 8 --lr 5e-5 --max_epoches 5 --max_steps 20000 --warmup_steps 100 --num_workers 2 --gpu_id 1 --k_folds 32/40

# python src/train.py --model_name_or_path $ENCODER --output_dir $OUTPUT_DIR --pretrained_cache_dir ./pretrained --batch_size 8 --lr 5e-5 --max_epoches 5 --max_steps 20000 --warmup_steps 100 --num_workers 2 --gpu_id 1 --k_folds 33/40

# python src/train.py --model_name_or_path $ENCODER --output_dir $OUTPUT_DIR --pretrained_cache_dir ./pretrained --batch_size 8 --lr 5e-5 --max_epoches 5 --max_steps 20000 --warmup_steps 100 --num_workers 2 --gpu_id 1 --k_folds 34/40

# python src/train.py --model_name_or_path $ENCODER --output_dir $OUTPUT_DIR --pretrained_cache_dir ./pretrained --batch_size 8 --lr 5e-5 --max_epoches 5 --max_steps 20000 --warmup_steps 100 --num_workers 2 --gpu_id 1 --k_folds 35/40

# python src/train.py --model_name_or_path $ENCODER --output_dir $OUTPUT_DIR --pretrained_cache_dir ./pretrained --batch_size 8 --lr 5e-5 --max_epoches 5 --max_steps 20000 --warmup_steps 100 --num_workers 2 --gpu_id 1 --k_folds 36/40

# python src/train.py --model_name_or_path $ENCODER --output_dir $OUTPUT_DIR --pretrained_cache_dir ./pretrained --batch_size 8 --lr 5e-5 --max_epoches 5 --max_steps 20000 --warmup_steps 100 --num_workers 2 --gpu_id 1 --k_folds 37/40

# python src/train.py --model_name_or_path $ENCODER --output_dir $OUTPUT_DIR --pretrained_cache_dir ./pretrained --batch_size 8 --lr 5e-5 --max_epoches 5 --max_steps 20000 --warmup_steps 100 --num_workers 2 --gpu_id 1 --k_folds 38/40

# python src/train.py --model_name_or_path $ENCODER --output_dir $OUTPUT_DIR --pretrained_cache_dir ./pretrained --batch_size 8 --lr 5e-5 --max_epoches 5 --max_steps 20000 --warmup_steps 100 --num_workers 2 --gpu_id 1 --k_folds 39/40