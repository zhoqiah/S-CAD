# S-CAD：多领域虚假新闻检测

**Semantic Cluster-Aware Asymmetric Disentanglement for Multi-Domain Fake News Detection**

S-CAD 是一个基于 PyTorch 的中文多领域虚假新闻检测研究项目，面向 Weibo21 数据集开展真假新闻二分类。实现结合预训练文本编码、词元图增强、多专家特征提取、注意力扩散和领域子空间建模，提供训练、验证与测试流程。


## 1. 方法与实现

主要计算流程为：

```text
新闻文本 + 领域类别
  → 中文 RoBERTa 编码（冻结参数）或预计算词向量
  → 可选 DADGNN 词元图增强
  → 领域感知门控与 5 个 CNN 专家
  → 注意力扩散 / 连续领域子空间及门控融合
  → MLP 二分类预测
```

| 模块 | 实现说明 | 主要文件 |
| --- | --- | --- |
| 文本编码 | 使用 BertModel 加载中文 RoBERTa；编码器参数冻结 | `models/scad_model.py` |
| 图增强 | 可选 DADGNN 分支，通过残差连接增强词元表示 | `models/dadgnn/` |
| 多专家建模 | 5 个 CNN 专家，结合领域嵌入进行门控加权 | `models/scad_model.py` |
| 特征处理 | 对池化表示和领域嵌入执行可选正交处理 | `models/orthographic.py` |
| 注意力扩散 | 在余弦相似度构建的词元图上迭代传播注意力 | `models/attention_diffusion.py` |
| 领域子空间 | 连续领域子空间与投影约束 | `models/domain_subspace.py` |
| 训练约束 | 二分类损失、领域对抗、领域分类、正交、投影及可选对比损失 | `models/scad_model.py` |

模型预测时也需要领域类别，当前接口并非仅输入新闻文本。具体方法以源码为准，本文不对其与论文的逐项一致性作验证。

## 2. 项目结构

以下为所提供源码的组织结构，数据和权重需要另行补齐：

```text
SCAD/
├── main.py                         # 命令行入口与配置
├── run.py                          # 数据加载及训练器调度
├── requirements.txt                # 原始依赖清单
├── README.md
├── data/
│   └── weibo21/                    # 默认数据位置（需准备）
│       ├── train.pkl
│       ├── val.pkl
│       └── test.pkl
├── pretrained_model/
│   ├── chinese_roberta_wwm_base_ext_pytorch/
│   │   ├── config.json
│   │   ├── vocab.txt
│   │   └── pytorch_model.bin        # 预训练权重（需补齐）
│   └── w2v/
│       └── Tencent_AILab_Chinese_w2v_model.kv
├── models/
│   ├── scad_model.py                # 实际入口使用的模型和 Trainer
│   ├── scad.py
│   ├── layers.py
│   ├── attention_diffusion.py
│   ├── domain_subspace.py
│   ├── orthographic.py
│   ├── sc_attention.py
│   └── dadgnn/
└── utils/
    ├── dataloader.py                # 数据读取、分词及张量构建
    └── utils.py                     # 指标、早停及设备辅助函数
```

本 README 保存于 SCAD 工作目录；所提供的源码目前位于另一个临时目录。运行前需将完整源码放在同一项目根目录，或进入原源码目录执行命令。

## 3. 环境与依赖

核心依赖包括 PyTorch、Transformers、DGL、NumPy、pandas、scikit-learn、gensim 和 jieba。

原始依赖清单包含 `torch==1.8.0`、`numpy==1.19.5`、`pandas==1.1.5`、`scikit-learn==0.24.1`、`gensim==4.0.1`、`transformers==4.13.0.dev0` 及 `dgl>=0.9.0`。这些是原始环境记录，不代表已验证的可安装环境。

在独立 Python 环境中、进入包含 requirements.txt 的源码根目录后，可按原项目方式安装：

```bash
python -m pip install -r requirements.txt
```

注意以下复现限制：

- 当前 `main.py` 将 `use_cuda` 固定为 True，需要可用的 CUDA GPU；没有直接提供 CPU 命令行开关。
- 依赖中存在开发版本及带构建后缀的包，安装可能需要匹配原始来源、Python 和 CUDA 环境，不能保证上述命令直接成功。
- `utils/dataloader.py` 导入了 `gensim.models.keyedvectors.Vocab`，需核对其与所安装 gensim 版本的兼容性。
- 尚未建立经过测试的 Python、CUDA、PyTorch 与 DGL 版本组合。

## 4. 数据准备

默认读取 Weibo21 的三个划分文件：

```text
data/weibo21/train.pkl
data/weibo21/val.pkl
data/weibo21/test.pkl
```

每个文件应保存一个可由 pickle 读取的 pandas DataFrame，至少包含以下列：

| 字段 | 内容 |
| --- | --- |
| `content` | 新闻文本字符串 |
| `label` | 可转为整数的二分类标签，取值为 0 或 1 |
| `category` | 与代码映射一致的中文领域名称 |

源码没有明确给出 0、1 与真假新闻的语义对应，使用前需核对数据标注，避免反向解释预测结果。

支持的领域及编号为：科技（0）、军事（1）、教育考试（2）、灾难事故（3）、政治（4）、医药健康（5）、财经商业（6）、文体娱乐（7）、社会生活（8）。加载器会过滤 `category` 为“无法确定”的样本，其他未知领域会导致映射失败。

自定义数据目录可用 `--root_path` 指定。由于代码通过字符串拼接生成文件路径，目录参数必须保留末尾的 `/`。

当前所提供材料没有包含上述三个数据文件，也没有提供可核验的数据获取说明。

## 5. 预训练模型准备

默认使用本地目录：

```text
pretrained_model/chinese_roberta_wwm_base_ext_pytorch/
```

该目录应包含兼容 BertModel 的配置、词表及完整模型权重。当前提供的目录可见 `config.json` 和 `vocab.txt`，尚缺默认加载所需的预训练权重。

原 README 提供了一个[预训练模型下载地址](https://drive.google.com/drive/folders/1y2k22iMG1i1f302NLf-bj7UEe9zwTwLR?usp=sharing)。该地址沿用原材料，本文未验证其可访问性或文件完整性。

可使用 `--bert` 和 `--bert_vocab_file` 分别指定模型目录与词表路径。

## 6. 训练与测试

在依赖、数据、模型权重和 CUDA 环境准备完毕后，从源码根目录执行：

```bash
python main.py --model_name mdfend --emb_type bert --batchsize 32 --lr 0.0007 --root_path ./data/weibo21/
```

**入口名称说明：** 项目名称为 S-CAD，但当前 `run.py` 只接受 `model_name == 'mdfend'`，并调用 `models/scad_model.py` 中的 Trainer。原 README 中的 `--model_name scad` 会触发 `Unknown model_name`。

使用默认配置训练：

```bash
python main.py
```

训练流程包含训练集优化、逐轮验证、按验证集宏平均 F1 保存最佳参数及早停，随后加载最佳参数进行测试。若没有生成检查点，代码会提示并使用最后一轮参数测试。

默认参数保存位置为：

```text
param_model/mdfend/parameter_mdfend.pkl
```

当前入口没有单独的预测或仅测试命令；重新执行 main.py 会进入训练流程。

## 7. 常用参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--model_name` | `mdfend` | 当前调度入口接受的模型名称 |
| `--epoch` | 50 | 最大训练轮数 |
| `--batchsize` | 64 | 批大小 |
| `--lr` | 0.0001 | 学习率 |
| `--max_len` | 170 | 最大序列长度 |
| `--early_stop` | 5 | 验证指标连续未提升的早停轮数 |
| `--seed` | 2026 | 主入口随机种子 |
| `--gpu` | `0` | CUDA_VISIBLE_DEVICES 设置 |
| `--num_workers` | 4 | 数据加载进程数 |
| `--emb_type` | `bert` | `bert` 或 `w2v` |
| `--save_param_dir` | `./param_model` | 参数保存根目录 |
| `--use_dadgnn` | 1 | 启用图增强 |
| `--use_ortho` | 1 | 启用正交特征处理 |
| `--use_scad_paper` | 1 | 启用扩散、子空间及相关约束路径 |
| `--use_grl` | 1 | 控制非 SCAD 路径的领域对抗分支 |
| `--use_sc_loss` | 0 | 启用辅助对比损失，需同时启用 DADGNN |
| `--adv_loss_weight` | 0.1 | 领域对抗损失权重 |
| `--dom_loss_weight` | 0.5 | 领域分类损失权重 |
| `--orth_loss_weight` | 0.01 | 正交损失权重 |
| `--proj_loss_weight` | 0.1 | 投影损失权重 |
| `--tau_diffusion` | 0.65 | 相似度图阈值 |
| `--gamma_diffusion` | 0.15 | 注意力扩散重启系数 |
| `--T_diffusion` | 5 | 扩散迭代次数 |

开关参数使用整数 0 / 1。完整参数定义见 main.py。启用 SCAD 路径时，该路径内部仍构造梯度反转及领域对抗模块，不能仅用 `--use_grl 0` 断言已关闭全部领域对抗。

消融示例：关闭 DADGNN 图增强。

```bash
python main.py --model_name mdfend --use_dadgnn 0
```

消融示例：关闭 SCAD 扩散与子空间路径。

```bash
python main.py --model_name mdfend --use_scad_paper 0
```

上述命令用于表达配置方式，尚未实际运行验证。

## 8. 评价指标与复现

评估函数计算整体及各领域的 AUC、宏平均 Precision、Recall、F1 和 Accuracy。整体宏平均 F1 对应结果字典中的 `metric`，用于选择最佳检查点。

主入口设置 Python、NumPy、PyTorch 随机种子及 cuDNN 确定性选项；数据加载 worker 中另有固定种子。报告实验时仍应记录硬件、完整依赖版本、数据划分、实际参数和多次运行结果，不能仅凭种子保证跨环境完全一致。

当前材料未提供训练日志或实验结果，因此本文不列出精度、性能提升或论文复现结论。

## 9. 常见问题

- **提示 Unknown model_name**：使用 `--model_name mdfend`。
- **找不到数据文件**：检查三个 pkl 文件及 `--root_path`，并保留目录末尾 `/`。
- **模型加载失败**：检查本地配置、词表及完整权重是否齐全且兼容。
- **CUDA 错误**：检查 GPU 可用性和依赖兼容性；CPU 运行需要调整源码配置。
- **显存不足**：尝试减小 batchsize 或 max_len，并记录修改，便于比较实验。
- **数据加载进程异常**：可尝试 `--num_workers 0` 辅助定位问题。

## 10. 许可证与引用

所提供材料未包含 LICENSE 或完整论文引用信息。正式发布时应补充项目许可证、数据与预训练模型的使用条款，以及经核实的论文信息；本文不编造作者、会议或 BibTeX 条目。
