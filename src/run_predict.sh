export ENCODER=hfl/chinese-roberta-wwm-ext

# python src/predict.py \
# --model_name_or_path $ENCODER \
# --pretrained_cache_dir ./pretrained \
# --batch_size 8 \
# --model_dir output/kfolds/0/step_2834

# python src/predict.py \
# --model_name_or_path $ENCODER \
# --pretrained_cache_dir ./pretrained \
# --batch_size 8 \
# --model_dir output/kfolds/1/step_2834

# python src/predict.py \
# --model_name_or_path $ENCODER \
# --pretrained_cache_dir ./pretrained \
# --batch_size 8 \
# --model_dir output/kfolds/2/step_5508

# python src/predict.py \
# --model_name_or_path $ENCODER \
# --pretrained_cache_dir ./pretrained \
# --batch_size 8 \
# --model_dir output/kfolds/3/step_2792

# python src/predict.py \
# --model_name_or_path $ENCODER \
# --pretrained_cache_dir ./pretrained \
# --batch_size 8 \
# --model_dir output/kfolds/4/step_7070

# python src/predict.py \
# --model_name_or_path $ENCODER \
# --pretrained_cache_dir ./pretrained \
# --batch_size 8 \
# --model_dir output/kfolds/5/step_5592

# python src/predict.py \
# --model_name_or_path $ENCODER \
# --pretrained_cache_dir ./pretrained \
# --batch_size 8 \
# --model_dir output/kfolds/6/step_2834

# python src/predict.py \
# --model_name_or_path $ENCODER \
# --pretrained_cache_dir ./pretrained \
# --batch_size 8 \
# --model_dir output/kfolds/7/step_7065

# python src/predict.py \
# --model_name_or_path $ENCODER \
# --pretrained_cache_dir ./pretrained \
# --batch_size 8 \
# --model_dir output/kfolds/8/step_7100

# python src/predict.py \
# --model_name_or_path $ENCODER \
# --pretrained_cache_dir ./pretrained \
# --batch_size 8 \
# --model_dir output/kfolds/9/step_2806

# 0
python src/predict.py \
--model_name_or_path $ENCODER \
--pretrained_cache_dir ./pretrained \
--batch_size 4 \
--model_dir output/kfolds_no_crf/10-16_21-10-31/step_2834


# 1
python src/predict.py \
--model_name_or_path $ENCODER \
--pretrained_cache_dir ./pretrained \
--batch_size 4 \
--model_dir output/kfolds_no_crf/10-16_22-50-47/step_4251


# 2
python src/predict.py \
--model_name_or_path $ENCODER \
--pretrained_cache_dir ./pretrained \
--batch_size 4 \
--model_dir output/kfolds_no_crf/10-17_00-29-10/step_5508


# 3
python src/predict.py \
--model_name_or_path $ENCODER \
--pretrained_cache_dir ./pretrained \
--batch_size 4 \
--model_dir output/kfolds_no_crf/10-17_02-06-04/step_4188


# 4
python src/predict.py \
--model_name_or_path $ENCODER \
--pretrained_cache_dir ./pretrained \
--batch_size 4 \
--model_dir output/kfolds_no_crf/10-17_03-46-38/step_4242


# 5
python src/predict.py \
--model_name_or_path $ENCODER \
--pretrained_cache_dir ./pretrained \
--batch_size 4 \
--model_dir output/kfolds_no_crf/10-17_05-25-53/step_2796


# 6
python src/predict.py \
--model_name_or_path $ENCODER \
--pretrained_cache_dir ./pretrained \
--batch_size 4 \
--model_dir output/kfolds_no_crf/10-17_07-05-49/step_7085


# 7
python src/predict.py \
--model_name_or_path $ENCODER \
--pretrained_cache_dir ./pretrained \
--batch_size 4 \
--model_dir output/kfolds_no_crf/10-17_08-45-55/step_7065


# 8
python src/predict.py \
--model_name_or_path $ENCODER \
--pretrained_cache_dir ./pretrained \
--batch_size 4 \
--model_dir output/kfolds_no_crf/10-17_10-24-00/step_7100


# 9
python src/predict.py \
--model_name_or_path $ENCODER \
--pretrained_cache_dir ./pretrained \
--batch_size 4 \
--model_dir output/kfolds_no_crf/10-17_12-04-21/step_4209
