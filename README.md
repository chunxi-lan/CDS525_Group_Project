# CDS525_Group_Project - CIFAR-100 图像分类

## 项目概述

本项目是CDS525深度学习课程的小组作业，使用CIFAR-100数据集进行图像分类任务。

## 项目结构

```
CDS525_Group_Project/
├── data/                    # 数据存放目录（自动创建）
├── checkpoints/             # 模型保存目录（自动创建）
├── results/                 # 结果保存目录（自动创建）
├── src/
│   ├── __init__.py
│   ├── data_loader.py       # 数据加载与预处理模块
│   ├── models.py            # 模型定义模块
│   ├── trainer.py           # 训练与验证核心逻辑
│   └── utils.py             # 工具函数（保存、日志等）
├── config.py                # 配置文件
├── train.py                 # 主训练脚本
└── requirements.txt         # 依赖包列表
```

## 组员分工

### 组员A：数据预处理 + 模型定义 + 基础训练框架（已完成）
- 数据加载与预处理
- 模型定义（ResNet-18/34/50）
- 训练与验证核心逻辑
- 主训练脚本

### 组员B：实验管理 + 超参数对比
- 批量运行实验（不同损失函数、学习率、batch size）
- 实验结果保存为JSON文件
- 自动化实验脚本

### 组员C：可视化 + 测试集结果展示
- 绘制所有规定图表
- 前100个测试结果可视化
- 代码整合与README编写

## 环境配置

### 安装依赖

```bash
pip install -r requirements.txt
```

### 依赖包列表
- torch>=2.0.0
- torchvision>=0.15.0
- numpy>=1.24.0
- matplotlib>=3.7.0
- scikit-learn>=1.3.0
- tqdm>=4.65.0

## 使用方法

### 1. 运行基准训练（默认参数）

```bash
python train.py
```

### 2. 自定义参数运行

```bash
python train.py --batch_size 32 --lr 0.001 --epochs 50 --model resnet18
```

### 3. 命令行参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| --batch_size | int | 64 | 批次大小 |
| --lr | float | 0.001 | 学习率 |
| --epochs | int | 50 | 训练轮数 |
| --model | str | resnet18 | 模型架构（resnet18/resnet34/resnet50） |
| --loss | str | cross_entropy | 损失函数 |
| --optimizer | str | adam | 优化器（adam/sgd） |
| --pretrained | flag | True | 使用预训练权重 |
| --freeze_backbone | flag | False | 冻结骨干网络 |
| --seed | int | 42 | 随机种子 |
| --checkpoint_name | str | best_model.pth | 模型保存文件名 |
| --metrics_name | str | None | 指标保存文件名（默认带时间戳） |

### 4. 批量实验示例（组员B参考）

```bash
# 不同学习率实验
python train.py --lr 0.1 --metrics_name metrics_lr0.1.json
python train.py --lr 0.01 --metrics_name metrics_lr0.01.json
python train.py --lr 0.001 --metrics_name metrics_lr0.001.json
python train.py --lr 0.0001 --metrics_name metrics_lr0.0001.json

# 不同batch size实验
python train.py --batch_size 8 --metrics_name metrics_bs8.json
python train.py --batch_size 16 --metrics_name metrics_bs16.json
python train.py --batch_size 32 --metrics_name metrics_bs32.json
python train.py --batch_size 64 --metrics_name metrics_bs64.json
python train.py --batch_size 128 --metrics_name metrics_bs128.json
```

## 输出文件说明

### 训练指标格式（JSON）
```json
{
    "train_loss": [2.5, 2.0, 1.5, ...],
    "train_acc": [30.0, 45.0, 55.0, ...],
    "val_loss": [...],
    "val_acc": [...],
    "test_loss": [...],
    "test_acc": [...]
}
```

### 模型检查点格式（.pth）
包含以下内容：
- model_state_dict: 模型状态字典
- optimizer_state_dict: 优化器状态字典
- scheduler_state_dict: 学习率调度器状态字典
- epoch: 训练轮数
- val_acc: 验证集准确率

## 数据集

- **数据集**: CIFAR-100
- **图像尺寸**: 32x32 RGB
- **类别数**: 100
- **训练集**: 50,000 张图像
- **测试集**: 10,000 张图像
- **下载方式**: 自动通过torchvision下载

## 模型架构

支持的模型：
- ResNet-18（默认）
- ResNet-34
- ResNet-50

所有模型均修改最后一层为100类输出，支持使用ImageNet预训练权重。

## 协作建议

1. **版本控制**: 使用GitHub组织代码，每位组员在自己的分支上开发
2. **接口约定**: 严格按照已定义的函数接口进行开发
3. **代码复用**: 组员B可直接调用train.py进行批量实验
4. **定期沟通**: 及时同步进度，解决依赖问题

