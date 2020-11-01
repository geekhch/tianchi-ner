# 解压数据并重命名
mkdir -p ./tmp_data/test1
cp /tcdata/juesai/* ./tmp_data/test1/
cp /data/juesai/* ./tmp_data/test1/
cp ./data/juesai/* ./tmp_data/test1/

export ENCODER=hfl/chinese-roberta-wwm-ext

python code/predict.py --model_name_or_path $ENCODER --data_dir tmp_data --pretrained_cache_dir ./user_data/pretrained --batch_size 4 --model_dir user_data/swa-nocrf-words-kfolds/0/epoch_5
python code/predict.py --model_name_or_path $ENCODER --data_dir tmp_data --pretrained_cache_dir ./user_data/pretrained --batch_size 4 --model_dir user_data/swa-nocrf-words-kfolds/1/epoch_5
python code/predict.py --model_name_or_path $ENCODER --data_dir tmp_data --pretrained_cache_dir ./user_data/pretrained --batch_size 4 --model_dir user_data/swa-nocrf-words-kfolds/2/epoch_5
python code/predict.py --model_name_or_path $ENCODER --data_dir tmp_data --pretrained_cache_dir ./user_data/pretrained --batch_size 4 --model_dir user_data/swa-nocrf-words-kfolds/3/epoch_5
python code/predict.py --model_name_or_path $ENCODER --data_dir tmp_data --pretrained_cache_dir ./user_data/pretrained --batch_size 4 --model_dir user_data/swa-nocrf-words-kfolds/4/epoch_5
python code/predict.py --model_name_or_path $ENCODER --data_dir tmp_data --pretrained_cache_dir ./user_data/pretrained --batch_size 4 --model_dir user_data/swa-nocrf-words-kfolds/5/epoch_5
python code/predict.py --model_name_or_path $ENCODER --data_dir tmp_data --pretrained_cache_dir ./user_data/pretrained --batch_size 4 --model_dir user_data/swa-nocrf-words-kfolds/6/epoch_5
python code/predict.py --model_name_or_path $ENCODER --data_dir tmp_data --pretrained_cache_dir ./user_data/pretrained --batch_size 4 --model_dir user_data/swa-nocrf-words-kfolds/7/epoch_5
python code/predict.py --model_name_or_path $ENCODER --data_dir tmp_data --pretrained_cache_dir ./user_data/pretrained --batch_size 4 --model_dir user_data/swa-nocrf-words-kfolds/8/epoch_5
python code/predict.py --model_name_or_path $ENCODER --data_dir tmp_data --pretrained_cache_dir ./user_data/pretrained --batch_size 4 --model_dir user_data/swa-nocrf-words-kfolds/9/epoch_5


python code/volt.py

rm result.zip

cd user_data/ && zip -r ../result.zip result && cd ..