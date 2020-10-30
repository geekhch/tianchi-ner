# 解压数据并重命名
mkdir -p ./tmp_data/test1
cp /tcdata/juesai/* ./tmp_data/test1/
cp /data/juesai/* ./tmp_data/test1/
cp ./data/juesai/* ./tmp_data/test1/

export ENCODER=hfl/chinese-roberta-wwm-ext

python code/predict.py --model_name_or_path $ENCODER --data_dir tmp_data --pretrained_cache_dir ./user_data/pretrained --batch_size 4 --model_dir user_data/kfolds_no_crf/step_4188
python code/predict.py --model_name_or_path $ENCODER --data_dir tmp_data --pretrained_cache_dir ./user_data/pretrained --batch_size 4 --model_dir user_data/kfolds_no_crf/step_2796
python code/predict.py --model_name_or_path $ENCODER --data_dir tmp_data --pretrained_cache_dir ./user_data/pretrained --batch_size 4 --model_dir user_data/kfolds_no_crf/step_4209
python code/predict.py --model_name_or_path $ENCODER --data_dir tmp_data --pretrained_cache_dir ./user_data/pretrained --batch_size 4 --model_dir user_data/kfolds_no_crf/step_4242
python code/predict.py --model_name_or_path $ENCODER --data_dir tmp_data --pretrained_cache_dir ./user_data/pretrained --batch_size 4 --model_dir user_data/kfolds_no_crf/step_7065
python code/predict.py --model_name_or_path $ENCODER --data_dir tmp_data --pretrained_cache_dir ./user_data/pretrained --batch_size 4 --model_dir user_data/kfolds_no_crf/step_5508
python code/predict.py --model_name_or_path $ENCODER --data_dir tmp_data --pretrained_cache_dir ./user_data/pretrained --batch_size 4 --model_dir user_data/kfolds_no_crf/step_2834
python code/predict.py --model_name_or_path $ENCODER --data_dir tmp_data --pretrained_cache_dir ./user_data/pretrained --batch_size 4 --model_dir user_data/kfolds_no_crf/step_7100
python code/predict.py --model_name_or_path $ENCODER --data_dir tmp_data --pretrained_cache_dir ./user_data/pretrained --batch_size 4 --model_dir user_data/kfolds_no_crf/step_4251
python code/predict.py --model_name_or_path $ENCODER --data_dir tmp_data --pretrained_cache_dir ./user_data/pretrained --batch_size 4 --model_dir user_data/kfolds_no_crf/step_7085
python code/predict.py --model_name_or_path $ENCODER --data_dir tmp_data --pretrained_cache_dir ./user_data/pretrained --batch_size 4 --model_dir user_data/kfolds/5/step_5592
python code/predict.py --model_name_or_path $ENCODER --data_dir tmp_data --pretrained_cache_dir ./user_data/pretrained --batch_size 4 --model_dir user_data/kfolds/3/step_2792
python code/predict.py --model_name_or_path $ENCODER --data_dir tmp_data --pretrained_cache_dir ./user_data/pretrained --batch_size 4 --model_dir user_data/kfolds/6/step_2834
python code/predict.py --model_name_or_path $ENCODER --data_dir tmp_data --pretrained_cache_dir ./user_data/pretrained --batch_size 4 --model_dir user_data/kfolds/4/step_7070
python code/predict.py --model_name_or_path $ENCODER --data_dir tmp_data --pretrained_cache_dir ./user_data/pretrained --batch_size 4 --model_dir user_data/kfolds/1/step_2834
python code/predict.py --model_name_or_path $ENCODER --data_dir tmp_data --pretrained_cache_dir ./user_data/pretrained --batch_size 4 --model_dir user_data/kfolds/9/step_2806
python code/predict.py --model_name_or_path $ENCODER --data_dir tmp_data --pretrained_cache_dir ./user_data/pretrained --batch_size 4 --model_dir user_data/kfolds/8/step_7100
python code/predict.py --model_name_or_path $ENCODER --data_dir tmp_data --pretrained_cache_dir ./user_data/pretrained --batch_size 4 --model_dir user_data/kfolds/2/step_5508
python code/predict.py --model_name_or_path $ENCODER --data_dir tmp_data --pretrained_cache_dir ./user_data/pretrained --batch_size 4 --model_dir user_data/kfolds/0/step_2834
python code/predict.py --model_name_or_path $ENCODER --data_dir tmp_data --pretrained_cache_dir ./user_data/pretrained --batch_size 4 --model_dir user_data/kfolds/7/step_7065


python code/volt.py

rm result.zip

cd user_data/ && zip -r ../result.zip result && cd ..