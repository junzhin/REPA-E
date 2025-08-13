# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述
REPA-E是一个用于端到端训练潜在扩散变换器的VAE模型。它实现了VAE与扩散模型的联合训练，通过表示对齐(REPA)损失实现稳定有效的训练。

## 核心创新点和具体实现位置

### 🎯 核心创新：投影对齐损失（Projection Alignment Loss）

**主要创新**：REPA-E的核心创新是通过投影对齐损失实现VAE和扩散模型的端到端训练。

#### 关键实现位置：

1. **投影对齐损失计算** - `loss/losses.py:411-422`：
   ```python
   proj_loss = torch.tensor(0., device=inputs.device)
   for i, (z, z_tilde) in enumerate(zip(zs, zs_tilde)):
       for j, (z_j, z_tilde_j) in enumerate(zip(z, z_tilde)):
           z_tilde_j = torch.nn.functional.normalize(z_tilde_j, dim=-1) 
           z_j = torch.nn.functional.normalize(z_j, dim=-1) 
           proj_loss += mean_flat(-(z_j * z_tilde_j).sum(dim=-1))
   ```

2. **SiT模型中的投影对齐** - `models/sit.py:372-379`：
   ```python
   proj_loss = torch.tensor(0., device=x.device)
   for i, (z, z_tilde) in enumerate(zip(zs, zs_tilde)):
       for z_j, z_tilde_j in zip(z, z_tilde):
           z_tilde_j = torch.nn.functional.normalize(z_tilde_j, dim=-1) 
           z_j = torch.nn.functional.normalize(z_j, dim=-1) 
           proj_loss += mean_flat(-(z_j * z_tilde_j).sum(dim=-1))
   ```

3. **端到端训练循环** - `train_repae.py:379-432`：
   - VAE对齐损失：`第389行`
   - SiT对齐损失：`第427行`

### 🔧 核心文件结构详解

#### 主要训练脚本
- **`train_repae.py`** - REPA-E端到端训练的核心脚本
  - 第379-398行：VAE训练和对齐损失计算
  - 第413-432行：SiT训练和投影对齐损失
  - 第225-246行：三个独立优化器（SiT、VAE、判别器）

- **`train_ldm_only.py`** - 仅训练扩散模型（固定VAE）的传统方法
- **`generate.py`** - 样本生成和评估脚本

#### 模型架构实现
- **`models/sit.py`** - SiT（Scalable Interpolant Transformers）扩散模型
  - 第308-388行：集成损失函数计算的前向传播
  - 第362-365行：投影器和对齐特征提取
  - 第23-30行：MLP投影器构建

- **`models/autoencoder.py`** - VAE模型实现，支持f8d4和f16d32架构
- **`models/`** - 包含多种视觉编码器（CLIP、DINOv2、MAE、MoCov3、JEPA）

#### 损失函数核心
- **`loss/losses.py`** - 损失函数集合
  - 第280-476行：`ReconstructionLoss_Single_Stage`类 - REPA-E的主要损失实现
  - 第378-475行：`_forward_generator_alignment`方法 - 投影对齐损失的核心实现
  - 第294-295行：投影系数参数定义

- **`loss/perceptual_loss.py`** - 感知损失（LPIPS）实现  
- **`loss/discriminator.py`** - GAN判别器实现

#### 数据和工具
- **`dataset.py`** - ImageNet-1K数据集处理
- **`preprocessing.py`** - 数据预处理脚本
- **`utils.py`** - 工具函数，包含编码器加载、特征归一化等
- **`samplers.py`** - 扩散采样策略实现

#### 辅助脚本
- **`cache_latents.py`** - E2E-VAE潜在表示缓存，用于加速训练
- **`save_vae_weights.py`** - 从REPA-E检查点提取VAE权重

## 环境设置
```bash
conda env create -f environment.yml -y
conda activate repa-e
```

## 主要训练命令

### 1. 数据预处理
```bash
python preprocessing.py --imagenet-path /PATH/TO/IMAGENET_TRAIN
```

### 2. REPA-E端到端训练（核心创新）
```bash
accelerate launch train_repae.py \
    --max-train-steps=400000 \
    --report-to="wandb" \
    --allow-tf32 \
    --mixed-precision="fp16" \
    --seed=0 \
    --data-dir="data" \
    --output-dir="exps" \
    --batch-size=256 \
    --path-type="linear" \
    --prediction="v" \
    --weighting="uniform" \
    --model="SiT-XL/2" \
    --checkpointing-steps=50000 \
    --loss-cfg-path="configs/l1_lpips_kl_gan.yaml" \
    --vae="f8d4" \
    --vae-ckpt="pretrained/sdvae/sdvae-f8d4.pt" \
    --disc-pretrained-ckpt="pretrained/sdvae/sdvae-f8d4-discriminator-ckpt.pt" \
    --enc-type="dinov2-vit-b" \
    --proj-coeff=0.5 \
    --encoder-depth=8 \
    --vae-align-proj-coeff=1.5 \
    --bn-momentum=0.1 \
    --exp-name="sit-xl-dinov2-b-enc8-repae-sdvae-0.5-1.5-400k"
```

### 3. E2E-VAE缓存潜在表示
```bash
accelerate launch --num_machines=1 --num_processes=8 cache_latents.py \
    --vae-arch="f16d32" \
    --vae-ckpt-path="pretrained/e2e-vavae/e2e-vavae-400k.pt" \
    --vae-latents-name="e2e-vavae" \
    --pproc-batch-size=128
```

### 4. 传统潜在扩散模型训练（固定VAE）
```bash
accelerate launch train_ldm_only.py \
    --max-train-steps=4000000 \
    --report-to="wandb" \
    --allow-tf32 \
    --mixed-precision="fp16" \
    --seed=0 \
    --data-dir="data" \
    --batch-size=256 \
    --path-type="linear" \
    --prediction="v" \
    --weighting="uniform" \
    --model="SiT-XL/1" \
    --checkpointing-steps=50000 \
    --vae="f16d32" \
    --vae-ckpt="pretrained/e2e-vavae/e2e-vavae-400k.pt" \
    --vae-latents-name="e2e-vavae" \
    --learning-rate=1e-4 \
    --output-dir="exps" \
    --exp-name="sit-xl-1-dinov2-b-enc8-ldm-only-e2e-vavae-0.5-4m"
```

### 5. 样本生成和评估
```bash
torchrun --nnodes=1 --nproc_per_node=8 generate.py \
    --num-fid-samples 50000 \
    --path-type linear \
    --mode sde \
    --num-steps 250 \
    --cfg-scale 1.0 \
    --guidance-high 1.0 \
    --guidance-low 0.0 \
    --exp-path pretrained/sit-ldm-e2e-vavae \
    --train-steps 4000000
```

### 6. 从REPA-E检查点提取VAE权重
```bash
python save_vae_weights.py \
    --repae-ckpt pretrained/sit-repae-vavae/checkpoints/0400000.pt \
    --vae-name e2e-vavae \
    --save-dir exps
```

## 核心技术架构

### REPA-E训练流程
1. **VAE训练阶段**：计算重建损失、感知损失、KL损失和VAE对齐损失
2. **判别器训练**：更新GAN判别器
3. **SiT训练阶段**：计算去噪损失和SiT投影对齐损失
4. **EMA更新**：更新SiT的指数移动平均参数

### 损失函数组合
- **重建损失**: L1损失用于像素级重建
- **感知损失**: LPIPS损失用于感知质量  
- **REPA损失**: 表示对齐损失（核心创新）
- **判别器损失**: GAN损失用于提升生成质量
- **KL散度**: 用于VAE正则化

### 支持的模型配置
- **VAE架构**: f8d4（8倍下采样，4通道）、f16d32（16倍下采样，32通道）
- **SiT模型**: SiT-B/L/XL，patch size 1/2
- **视觉编码器**: DINOv2、DINOv1、CLIP、MoCov3、MAE、JEPA

### 核心参数
- `--proj-coeff`: SiT投影对齐损失系数（通常0.5）
- `--vae-align-proj-coeff`: VAE投影对齐损失系数（通常1.5）
- `--encoder-depth`: 编码器深度，控制特征提取层数

## 预训练模型目录结构
```
pretrained/
├── sdvae/           # SD-VAE模型
├── invae/           # IN-VAE模型  
├── vavae/           # VA-VAE模型
├── e2e-sdvae/       # E2E调优的SD-VAE
├── e2e-invae/       # E2E调优的IN-VAE
└── e2e-vavae/       # E2E调优的VA-VAE

data/                # 预处理后的训练数据
exps/                # 实验输出目录
```