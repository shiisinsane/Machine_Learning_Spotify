## 项目结构

```
data_code/
├── split_train_test.py # 划分训练集与预测集
├── feature_process.py # 数据清洗和特征工程
├── description.py # 描述性统计
├── Modelling_final.py # 模型训练与预测
├── poisoning.py # 数据投毒，检验鲁棒性
├── result.py # 分析四种模型的性能并绘图
├── experiment_results/ # 结果输出目录（模型预测结果、评估报告、实验结果图等）
├── spotify-data.csv # 原始数据集
├── spotify_train(test).csv # 初始划分的训练集（测试集）
├── train(test)_processed.csv # 特征工程处理后的训练集（测试集），直接用于建模
└── README.md # 说明文档
```

## 运行说明

1. 数据准备：将`spotify-data.csv`与`split_train_test.py`放入同一目录下，运行得到`spotify_train(test).csv`
2. 特征工程：同一目录下，运行`feature_process.py`，得到`train(test)_processed.csv`
3. 描述性统计：同一目录下，运行`description.py`
4. 模型训练与预测：同一目录下，运行`Modelling_final.py`
5. 鲁棒性分析：同一目录下，运行`poisoning.py`
6. 结果可视化：同一目录下，运行`result.py`
