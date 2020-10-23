# 解压数据并重命名
unzip data/round1_test.zip -d data/ && mv data/chusai_xuanshou data/test1

export ENCODER=hfl/chinese-roberta-wwm-ext

python code/predict.py \
--model_name_or_path $ENCODER \
--pretrained_cache_dir ./user_data/pretrained \
--batch_size 8 \
--model_dir user_data/kfolds/step_2834

python code/predict.py \
--model_name_or_path $ENCODER \
--pretrained_cache_dir ./user_data/pretrained \
--batch_size 8 \
--model_dir user_data/kfolds/step_2834

python code/predict.py \
--model_name_or_path $ENCODER \
--pretrained_cache_dir ./user_data/pretrained \
--batch_size 8 \
--model_dir user_data/kfolds/step_5508

python code/predict.py \
--model_name_or_path $ENCODER \
--pretrained_cache_dir ./user_data/pretrained \
--batch_size 8 \
--model_dir user_data/kfolds/step_2792

python code/predict.py \
--model_name_or_path $ENCODER \
--pretrained_cache_dir ./user_data/pretrained \
--batch_size 8 \
--model_dir user_data/kfolds/step_7070

python code/predict.py \
--model_name_or_path $ENCODER \
--pretrained_cache_dir ./user_data/pretrained \
--batch_size 8 \
--model_dir user_data/kfolds/step_5592

python code/predict.py \
--model_name_or_path $ENCODER \
--pretrained_cache_dir ./user_data/pretrained \
--batch_size 8 \
--model_dir user_data/kfolds/step_2834

python code/predict.py \
--model_name_or_path $ENCODER \
--pretrained_cache_dir ./user_data/pretrained \
--batch_size 8 \
--model_dir user_data/kfolds/step_7065

python code/predict.py \
--model_name_or_path $ENCODER \
--pretrained_cache_dir ./user_data/pretrained \
--batch_size 8 \
--model_dir user_data/kfolds/step_7100

python code/predict.py \
--model_name_or_path $ENCODER \
--pretrained_cache_dir ./user_data/pretrained \
--batch_size 8 \
--model_dir user_data/kfolds/step_2806

# 0
python code/predict.py \
--model_name_or_path $ENCODER \
--pretrained_cache_dir ./user_data/pretrained \
--batch_size 4 \
--model_dir user_data/kfolds_no_crf/10-16_21-10-31/step_2834


# 1
python code/predict.py \
--model_name_or_path $ENCODER \
--pretrained_cache_dir ./user_data/pretrained \
--batch_size 4 \
--model_dir user_data/kfolds_no_crf/10-16_22-50-47/step_4251


# 2
python code/predict.py \
--model_name_or_path $ENCODER \
--pretrained_cache_dir ./user_data/pretrained \
--batch_size 4 \
--model_dir user_data/kfolds_no_crf/10-17_00-29-10/step_5508


# 3
python code/predict.py \
--model_name_or_path $ENCODER \
--pretrained_cache_dir ./user_data/pretrained \
--batch_size 4 \
--model_dir user_data/kfolds_no_crf/10-17_02-06-04/step_4188


# 4
python code/predict.py \
--model_name_or_path $ENCODER \
--pretrained_cache_dir ./user_data/pretrained \
--batch_size 4 \
--model_dir user_data/kfolds_no_crf/10-17_03-46-38/step_4242


# 5
python code/predict.py \
--model_name_or_path $ENCODER \
--pretrained_cache_dir ./user_data/pretrained \
--batch_size 4 \
--model_dir user_data/kfolds_no_crf/10-17_05-25-53/step_2796


# 6
python code/predict.py \
--model_name_or_path $ENCODER \
--pretrained_cache_dir ./user_data/pretrained \
--batch_size 4 \
--model_dir user_data/kfolds_no_crf/10-17_07-05-49/step_7085


# 7
python code/predict.py \
--model_name_or_path $ENCODER \
--pretrained_cache_dir ./user_data/pretrained \
--batch_size 4 \
--model_dir user_data/kfolds_no_crf/10-17_08-45-55/step_7065


# 8
python code/predict.py \
--model_name_or_path $ENCODER \
--pretrained_cache_dir ./user_data/pretrained \
--batch_size 4 \
--model_dir user_data/kfolds_no_crf/10-17_10-24-00/step_7100


# 9
python code/predict.py \
--model_name_or_path $ENCODER \
--pretrained_cache_dir ./user_data/pretrained \
--batch_size 4 \
--model_dir user_data/kfolds_no_crf/10-17_12-04-21/step_4209

python code/volt.py

rm prediction_result/* && cd user_data/ && zip -r ../prediction_result/result.zip result && cd ..