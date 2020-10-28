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

python code/predict.py --model_name_or_path $ENCODER --data_dir tmp_data --pretrained_cache_dir ./user_data/pretrained --batch_size 4 --model_dir user_data/kfolds_no_crf_avged/10-17_03-46-38/merge_model
python code/predict.py --model_name_or_path $ENCODER --data_dir tmp_data --pretrained_cache_dir ./user_data/pretrained --batch_size 4 --model_dir user_data/kfolds_no_crf_avged/10-17_08-45-55/merge_model
python code/predict.py --model_name_or_path $ENCODER --data_dir tmp_data --pretrained_cache_dir ./user_data/pretrained --batch_size 4 --model_dir user_data/kfolds_no_crf_avged/10-16_21-10-31/merge_model
python code/predict.py --model_name_or_path $ENCODER --data_dir tmp_data --pretrained_cache_dir ./user_data/pretrained --batch_size 4 --model_dir user_data/kfolds_no_crf_avged/10-17_00-29-10/merge_model
python code/predict.py --model_name_or_path $ENCODER --data_dir tmp_data --pretrained_cache_dir ./user_data/pretrained --batch_size 4 --model_dir user_data/kfolds_no_crf_avged/10-17_07-05-49/merge_model
python code/predict.py --model_name_or_path $ENCODER --data_dir tmp_data --pretrained_cache_dir ./user_data/pretrained --batch_size 4 --model_dir user_data/kfolds_no_crf_avged/10-17_10-24-00/merge_model
python code/predict.py --model_name_or_path $ENCODER --data_dir tmp_data --pretrained_cache_dir ./user_data/pretrained --batch_size 4 --model_dir user_data/kfolds_no_crf_avged/10-17_05-25-53/merge_model
python code/predict.py --model_name_or_path $ENCODER --data_dir tmp_data --pretrained_cache_dir ./user_data/pretrained --batch_size 4 --model_dir user_data/kfolds_no_crf_avged/10-17_12-04-21/merge_model
python code/predict.py --model_name_or_path $ENCODER --data_dir tmp_data --pretrained_cache_dir ./user_data/pretrained --batch_size 4 --model_dir user_data/kfolds_no_crf_avged/10-17_02-06-04/merge_model
python code/predict.py --model_name_or_path $ENCODER --data_dir tmp_data --pretrained_cache_dir ./user_data/pretrained --batch_size 4 --model_dir user_data/kfolds_no_crf_avged/10-16_22-50-47/merge_model

python code/volt.py

rm result.zip

cd user_data/ && zip -r ../result.zip result && cd ..