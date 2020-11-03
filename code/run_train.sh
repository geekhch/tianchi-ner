

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
# export ENCODER=voidful/albert_chinese_small

export OUTPUT_DIR=./output/swa-nocrf-kfolds
# # export ENCODER=allenyummy/chinese-bert-wwm-ehr-ner-sl

python code/train.py --model_name_or_path $ENCODER --output_dir $OUTPUT_DIR --pretrained_cache_dir ./pretrained --batch_size 8 --lr 5e-5 --max_epoches 9 --max_steps 20000 --warmup_steps 100 --num_workers 1 --gpu_id 1 --k_folds 0/10
python code/train.py --model_name_or_path $ENCODER --output_dir $OUTPUT_DIR --pretrained_cache_dir ./pretrained --batch_size 8 --lr 5e-5 --max_epoches 9 --max_steps 20000 --warmup_steps 100 --num_workers 1 --gpu_id 1 --k_folds 1/10
python code/train.py --model_name_or_path $ENCODER --output_dir $OUTPUT_DIR --pretrained_cache_dir ./pretrained --batch_size 8 --lr 5e-5 --max_epoches 9 --max_steps 20000 --warmup_steps 100 --num_workers 1 --gpu_id 1 --k_folds 2/10
python code/train.py --model_name_or_path $ENCODER --output_dir $OUTPUT_DIR --pretrained_cache_dir ./pretrained --batch_size 8 --lr 5e-5 --max_epoches 9 --max_steps 20000 --warmup_steps 100 --num_workers 1 --gpu_id 1 --k_folds 3/10
python code/train.py --model_name_or_path $ENCODER --output_dir $OUTPUT_DIR --pretrained_cache_dir ./pretrained --batch_size 8 --lr 5e-5 --max_epoches 9 --max_steps 20000 --warmup_steps 100 --num_workers 1 --gpu_id 1 --k_folds 4/10
python code/train.py --model_name_or_path $ENCODER --output_dir $OUTPUT_DIR --pretrained_cache_dir ./pretrained --batch_size 8 --lr 5e-5 --max_epoches 9 --max_steps 20000 --warmup_steps 100 --num_workers 1 --gpu_id 1 --k_folds 5/10
python code/train.py --model_name_or_path $ENCODER --output_dir $OUTPUT_DIR --pretrained_cache_dir ./pretrained --batch_size 8 --lr 5e-5 --max_epoches 9 --max_steps 20000 --warmup_steps 100 --num_workers 1 --gpu_id 1 --k_folds 6/10
python code/train.py --model_name_or_path $ENCODER --output_dir $OUTPUT_DIR --pretrained_cache_dir ./pretrained --batch_size 8 --lr 5e-5 --max_epoches 9 --max_steps 20000 --warmup_steps 100 --num_workers 1 --gpu_id 1 --k_folds 7/10
python code/train.py --model_name_or_path $ENCODER --output_dir $OUTPUT_DIR --pretrained_cache_dir ./pretrained --batch_size 8 --lr 5e-5 --max_epoches 9 --max_steps 20000 --warmup_steps 100 --num_workers 1 --gpu_id 1 --k_folds 8/10
python code/train.py --model_name_or_path $ENCODER --output_dir $OUTPUT_DIR --pretrained_cache_dir ./pretrained --batch_size 8 --lr 5e-5 --max_epoches 9 --max_steps 20000 --warmup_steps 100 --num_workers 1 --gpu_id 1 --k_folds 9/10
