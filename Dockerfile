# Base Images
## 指定了版
FROM registry.cn-shenzhen.aliyuncs.com/tianchi-tcm/tianchi-sz:nocrf

# RUN mkdir /tianchi

## 把当前文件夹里的文件构建到镜像的根目录下

# COPY ./code/run_predict.sh /tianchi/code/run_predict.sh
COPY . /tianchi/

## 指定默认工作目录为根目录（需要把run.sh和生成的结果文件都放在该文件夹下，提交后才能运行）
WORKDIR /tianchi/

## 镜像启动后统一执行 sh run.sh
CMD ["sh", "run.sh"]q
