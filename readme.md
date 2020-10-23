# 1 解决方案及算法
## 算法在BERT、RoBERTa预训练模型基础上，针对训练集数据进行微调，同时在模型上层应用条件随机场模型，通过训练转移矩阵加强序列标注的正确性。最终采用的最优模型分别对BERT-CRF、BERT、RoBERTa-CRF、RoBERTa四个模型进行10折交叉验证训练得到40个模型。使用40个模型进行集成投票得到最终结果。

# 2 依赖包版本
> * PYTHON：3.6.5
> * PYTORCH：1.6.0
> * CUDA：11.0
> * CUDNN：10.2

# 3 提交说明
## 此算法使用了4组10折交叉验证进行模型集成，其中2组基于bert-chinese-base预训练模型进行微调。由于上传限制，仅上传了基于bert-chinese-base模型的预测结果，存放于merge_5e-5.pkl文件中。通过运行.sh文件，此pkl文件可以结合已上传的20个RoBERTa模型进行预测和结果复现。

# 4 运行说明
项目根目录下运行`bash code/run_predict.sh`