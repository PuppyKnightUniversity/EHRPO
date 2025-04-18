# Hitanet Modified Version

## Project information
本文件夹基于pyhealth实现了hitanet的修改版本。

对每一个visit内部，不再使用线性层进行编码，而是采用transformer实现。


实现code attention和visit attention的提取。

实现小模型指导的简单out come reward，增强LLM的EHR分析能力。


LLM层面对feature level attention以及visit level attention的分布抽取&观察


多增加任务，除了mortality predition，再加一个再入院率预测会好一些

## Train model
1.激活环境：
```bash
conda activate lame
```
2.训练模型：
```bash
python trainer_hitanet.py 
```