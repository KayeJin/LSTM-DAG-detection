# 总览
本代码基于 https://arxiv.org/abs/1611.00791 “Predicting Domain Generation Algorithms with Long Short-Term Memory Networks” 修改 

其中：

    python版本：3.8.19

    benign域名数据集来源： https://github.com/PeterDaveHello/top-1m-domains 

    malicous域名数据集来源：Kaggle https://www.kaggle.com/datasets/gtkcyber/dga-dataset?resource=download 或者https://github.com/chrmor/DGA_domains_dataset

    除了现有数据集，也可以使用DGA算法生成： https://github.com/baderj/domain_generation_algorithms
    
    运行代码：python run.py 

# 文件说明

lstm_model1.h5 是保存的lstm模型。

test.py 用于测试lstm模型，其中字符编码为整数的字典需要与lstm中使用的字典相同。编码文件在encoding.txt中
