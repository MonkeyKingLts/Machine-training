# CycleGAN 夏季 ↔ 冬季风景风格转换

基于 [CycleGAN](https://junyanz.github.io/CycleGAN/) 的无配对图像到图像翻译实现，支持两种视觉风格之间的**双向**转换，例如：

- **A → B**：夏季风景 → 冬季风景
- **B → A**：冬季风景 → 夏季风景

## 项目结构

```
.
├── models/                 # 生成器、判别器与 CycleGAN 封装
├── datasets/               # 无配对双域数据集
├── scripts/
│   ├── download_dataset.py # 下载官方 Yosemite 数据集
│   └── generate_demo_data.py # 生成合成演示数据
├── train.py                # 训练脚本
├── test.py                 # 推理脚本
├── config.py               # 命令行配置
├── utils.py                # 工具函数
└── requirements.txt
```

## 环境安装

```bash
pip install -r requirements.txt
```

建议使用 Python 3.10+ 与 CUDA GPU 进行训练。

## 数据准备

### 方式一：官方 Summer↔Winter Yosemite 数据集（推荐）

```bash
python scripts/download_dataset.py --output-dir datasets/summer2winter_yosemite
```

解压后目录结构：

```
datasets/summer2winter_yosemite/
├── trainA/   # 夏季训练集
├── trainB/   # 冬季训练集
├── testA/
└── testB/
```

### 方式二：自定义无配对数据

将两个域的图像分别放入 `trainA/` 与 `trainB/`，**无需一一对应**：

```
your_dataset/
├── trainA/   # 域 A（如夏季）
└── trainB/   # 域 B（如冬季）
```

### 方式三：快速演示（合成数据）

```bash
python scripts/generate_demo_data.py --output-dir datasets/demo_summer_winter --count 32
```

## 训练

```bash
python train.py \
  --dataroot datasets/summer2winter_yosemite \
  --name summer2winter \
  --epochs 200 \
  --batch-size 1 \
  --image-size 256
```

常用参数：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--dataroot` | 数据集根目录 | 必填 |
| `--epochs` | 训练轮数 | 200 |
| `--lambda-cycle` | 循环一致性损失权重 | 10.0 |
| `--lambda-identity` | 恒等映射损失权重 | 0.5 |
| `--no-identity` | 关闭恒等损失 | 关闭 |
| `--resume` | 从检查点恢复 | 无 |

训练过程中：

- 检查点：`checkpoints/<name>/epoch_XXXX.pth`
- 可视化样本：`samples/<name>/epoch_XXXX.png`（展示 real_A → fake_B → rec_A 与 real_B → fake_A → rec_B）

## 推理

```bash
python test.py \
  --dataroot datasets/summer2winter_yosemite \
  --checkpoint checkpoints/summer2winter/epoch_0200.pth \
  --direction both \
  --num-images 10 \
  --output-dir results/summer2winter
```

`--direction` 可选：

- `AtoB`：夏季 → 冬季
- `BtoA`：冬季 → 夏季
- `both`：双向转换

输出为左右对比图（原图 | 转换结果）。

## 模型原理

CycleGAN 由两个生成器与两个判别器组成：

```
G_A: 域 A → 域 B    G_B: 域 B → 域 A
D_A: 判别域 A       D_B: 判别域 B
```

核心损失：

1. **对抗损失**：使生成图像在目标域中逼真
2. **循环一致性损失**：`A → B → A` 与 `B → A → B` 应接近原图
3. **恒等映射损失**（可选）：当目标域图像输入对应生成器时，输出应接近原图

生成器采用 ResNet 结构（9 个残差块），判别器采用 PatchGAN。

## 快速验证（演示数据 + 短训练）

```bash
python scripts/generate_demo_data.py
python train.py --dataroot datasets/demo_summer_winter --name demo --epochs 2 --save-interval 1
python test.py --dataroot datasets/demo_summer_winter --checkpoint checkpoints/demo/epoch_0002.pth --direction both
```

## 参考

- Zhu, J.-Y., et al. *Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks.* ICCV 2017.
- 官方实现：https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
