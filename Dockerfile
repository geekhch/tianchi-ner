# Base Images
## 指定了版
FROM registry.cn-shenzhen.aliyuncs.com/tianchi-tcm/tianchi-sz:new_models

RUN mkdir -p /tianchi/user_data

## 把当前文件夹里的文件构建到镜像的根目录下

# COPY ./code/run_predict.sh /tianchi/code/run_predict.sh
COPY . /tianchi/
# COPY ./user_data /tianchi/user_data/

## 指定默认工作目录为根目录（需要把run.sh和生成的结果文件都放在该文件夹下，提交后才能运行）
WORKDIR /tianchi/

## 镜像启动后统一执行 sh run.sh
CMD ["sh", "run.sh"]
