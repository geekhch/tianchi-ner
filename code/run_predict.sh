# 解压数据并重命名
mkdir -p ./tmp_data/test1
cp /tcdata/juesai/* ./tmp_data/test1/
cp /data/juesai/* ./tmp_data/test1/
cp ./data/juesai/* ./tmp_data/test1/

export ENCODER=hfl/chinese-roberta-wwm-ext

# python code/predict.py \
# --model_name_or_path $ENCODER \
# --pretrained_cache_dir ./user_data/pretrained \
# --batch_size 8 \
# --model_dir user_data/kfolds/step_2834

# python code/predict.py \
# --model_name_or_path $ENCODER \
# --pretrained_cache_dir ./user_data/pretrained \
# --batch_size 8 \
# --model_dir user_data/kfolds/step_2834

# python code/predict.py \
# --model_name_or_path $ENCODER \
# --pretrained_cache_dir ./user_data/pretrained \
# --batch_size 8 \
# --model_dir user_data/kfolds/step_5508

# python code/predict.py \
# --model_name_or_path $ENCODER \
# --pretrained_cache_dir ./user_data/pretrained \
# --batch_size 8 \
# --model_dir user_data/kfolds/step_2792

# python code/predict.py \
# --model_name_or_path $ENCODER \
# --pretrained_cache_dir ./user_data/pretrained \
# --batch_size 8 \
# --model_dir user_data/kfolds/step_7070

# python code/predict.py \
# --model_name_or_path $ENCODER \
# --pretrained_cache_dir ./user_data/pretrained \
# --batch_size 8 \
# --model_dir user_data/kfolds/step_5592

# python code/predict.py \
# --model_name_or_path $ENCODER \
# --pretrained_cache_dir ./user_data/pretrained \
# --batch_size 8 \
# --model_dir user_data/kfolds/step_2834

# python code/predict.py \
# --model_name_or_path $ENCODER \
# --pretrained_cache_dir ./user_data/pretrained \
# --batch_size 8 \
# --model_dir user_data/kfolds/step_7065

# python code/predict.py \
# --model_name_or_path $ENCODER \
# --pretrained_cache_dir ./user_data/pretrained \
# --batch_size 8 \
# --model_dir user_data/kfolds/step_7100

# python code/predict.py \
# --model_name_or_path $ENCODER \
# --pretrained_cache_dir ./user_data/pretrained \
# --batch_size 8 \
# --model_dir user_data/kfolds/step_2806

# 0
python code/predict.py \
--model_name_or_path $ENCODER \
--data_dir tmp_data \
--pretrained_cache_dir ./user_data/pretrained \
--batch_size 4 \
--model_dir user_data/kfolds_no_crf/step_2796


# 1
python code/predict.py \
--model_name_or_path $ENCODER \
--data_dir tmp_data \
--pretrained_cache_dir ./user_data/pretrained \
--batch_size 4 \
--model_dir user_data/kfolds_no_crf/step_2834


# 2
python code/predict.py \
--model_name_or_path $ENCODER \
--data_dir tmp_data \
--pretrained_cache_dir ./user_data/pretrained \
--batch_size 4 \
--model_dir user_data/kfolds_no_crf/step_4188


# 3
python code/predict.py \
--model_name_or_path $ENCODER \
--data_dir tmp_data \
--pretrained_cache_dir ./user_data/pretrained \
--batch_size 4 \
--model_dir user_data/kfolds_no_crf/step_4209


# 4
python code/predict.py \
--model_name_or_path $ENCODER \
--data_dir tmp_data \
--pretrained_cache_dir ./user_data/pretrained \
--batch_size 4 \
--model_dir user_data/kfolds_no_crf/step_4242


# 5
python code/predict.py \
--model_name_or_path $ENCODER \
--data_dir tmp_data \
--pretrained_cache_dir ./user_data/pretrained \
--batch_size 4 \
--model_dir user_data/kfolds_no_crf/step_4251


# 6
python code/predict.py \
--model_name_or_path $ENCODER \
--data_dir tmp_data \
--pretrained_cache_dir ./user_data/pretrained \
--batch_size 4 \
--model_dir user_data/kfolds_no_crf/step_5508


# 7
python code/predict.py \
--model_name_or_path $ENCODER \
--data_dir tmp_data \
--pretrained_cache_dir ./user_data/pretrained \
--batch_size 4 \
--model_dir user_data/kfolds_no_crf/step_7065


# 8
python code/predict.py \
--model_name_or_path $ENCODER \
--data_dir tmp_data \
--pretrained_cache_dir ./user_data/pretrained \
--batch_size 4 \
--model_dir user_data/kfolds_no_crf/step_7085


# 9
python code/predict.py \
--model_name_or_path $ENCODER \
--data_dir tmp_data \
--pretrained_cache_dir ./user_data/pretrained \
--batch_size 4 \
--model_dir user_data/kfolds_no_crf/step_7100

python code/volt.py

rm result.zip

cd user_data/ && zip -r ../result.zip result && cd ..