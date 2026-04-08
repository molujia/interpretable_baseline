# CLLM v8 — 可解释因果根因定位框架

---

## 快速开始

```bash
pip install -r requirements.txt

# 纯 CF-CBN（不调用 LLM）
python main.py --dataset ds25

# 启用 LLM（Workflow A + B，置信度触发）
python main.py --dataset ds25 --use-llm

# 使用 RAG 自适应 alpha
python main.py --dataset ds25 --use-llm --alpha-strategy rag

# 单 case 详细分析
python main.py --single 42 --use-llm --llm-mode always --verbose
```

---

## 系统架构

```
              Explainability Layer（LLM — 不在 RCL 关键路径上）
  告警 ──►  EdgeFilter · MetricClassifier · Anonymizer 屏障
            Explainer WF-A（SRE 确认前）· WF-B（SRE 确认后）
                           │
                    （清洗后的异常证据）
                           │
              Interpretability Layer（零 LLM，完全可审计）
            CF Engine ──► CBN Accumulator (alpha-tuner) ──► Case Store
                           │
                    Top-K 排名 + 英文故障工单
```

### 三个含 LLM 的模块

| 模块 | 调用时机 | 职责 |
|---|---|---|
| **EdgeFilter** | 系统初始化（第一个 case 进入之前） | Step-0 脚本删除先验知识中的虚假边；Step-2 LLM 进行 collider 偏差消除 |
| **MetricClassifier** | 系统初始化 | LLM 将语义等价指标合并为一个类；控制台打印合并组 |
| **Explainer** | CF-CBN 排名完成后 | WF-A：语义解读；WF-B：失误诊断 |

> **为何 EdgeFilter 和 MetricClassifier 在不加 `--use-llm` 时也会调用 LLM？**
>
> `--use-llm` 参数只控制 **每个 case** 是否触发 Explainer (WF-A/WF-B)。
> EdgeFilter 和 MetricClassifier 在系统启动时初始化，与该参数无关，原因如下：
>
> 1. 先验知识（虚假边判断、指标语义相似度判断）无法完全用脚本规则穷举；
> 2. 先验知识随业务变化而更新，LLM 可在不修改代码的情况下即时反映新知识；
> 3. 金丝雀部署、紧急重构等场景需要快速适应，等不了代码发布周期。
>
> 这两个 LLM 调用只在启动时发生一次，不在 RCL 关键路径上，不影响推理延迟。

---

## 自适应 Alpha 策略

融合公式：`score(s) = α·CF(s) + (1-α)·CBN(s)`

| 策略 | 参数 | 机制 |
|---|---|---|
| **Adaptive**（默认） | `adaptive` | 每 30 个 case 在最近 60 个 case 上做 LOO 二分搜索 |
| **RAG (BoW)** | `rag` | `α = 1 − cosine_sim(新证据, 最相似历史)` — 本地 BoW，无需 API |
| **RAG (API)** | `rag_api` | 与 `rag` 原理相同，但向量化使用 APIYI text-embedding-3-small（1536维） |
| **Jaccard** | `jaccard` | `α = 1 − max_weighted_jaccard(新 case 特征, 历史特征)` |
| **Entropy** | `entropy` | `α = α_min + (α_max−α_min)·H_norm(CBN 后验分布熵)` |

### 切换 RAG 向量化后端

打开 `cfcbn/rag_case_store.py`，修改文件顶部的开关：

```python
# 本地 BoW 编码器（默认，无需 API Key）
USE_OPENAI_EMBED = False

# OpenAI text-embedding-3-small（需要 Key）
USE_OPENAI_EMBED = True
OPENAI_API_KEY   = "sk-your-key-here"   # 填入你的 Key
```

切换后，删除 `case_store/rag_index.faiss`、`rag_vocab.pkl`、`rag_meta.pkl`，
让向量库用新后端重建。

---

## 故障工单格式（全英文）

```
======================================================================
  CLLM v5 -- Fault Analysis Ticket
  Fault category : jvm fault  |  Fault type: jvm gc
======================================================================

[ 1 ]  CF-CBN Root-Cause Ranking
  #1  adservice                       fused_score=1.0000
       [alpha=1.000 [adaptive]]

[ 2 ]  Anomaly Evidence  (判断依据：候选服务的原始异常指标)
  Top candidates:
    adservice     ['pod_cpu_usage', 'rrt', 'server_error']  (CF_raw=0.3821)
    frontend      ['client_error', 'rrt']  (CF_raw=0.2104)

[ 3 ]  LLM Workflow A -- Semantic Interpretation
  Trigger reason : C1: low confidence gap ...

[ 4 ]  Similar Historical Cases
----------------------------------------------------------------------
  ACTION  ...
```

---

## 消融实验

```bash
python ablation/ablation_run.py --dataset ds25 --group <GROUP>
```

| Group | 内容 |
|---|---|
| `A` | Alpha 策略对比（9 条曲线）：fixed α=1.0/0.5/0.7、v4 指数衰减、v5 自适应、RAG(BoW)、RAG(API)、Jaccard、Entropy |
| `B` | eval_window 敏感性 |
| `C` | EdgeFilter 拓扑噪声消融 |
| `D` | MetricClassifier 指标噪声消融 |
| `CD` | **联合消融**：EdgeFilter + MetricClassifier（4 种配置） |
| `all` | 运行全部组 |

### Group CD 配置说明

| 配置名 | EdgeFilter | MetricClassifier |
|---|---|---|
| `CD-full` | 启用 | 启用 |
| `CD-no-ef` | 禁用 | 启用 |
| `CD-no-mc` | 启用 | 禁用 |
| `CD-neither` | 禁用 | 禁用 |

> `--group CD` 同样会调用 `EdgeFilterAgent.run()`（Step-0 脚本 + Step-2 LLM），
> 与 `--group C` 行为一致，确保 LLM 辅助的 collider 消除步骤被完整执行。

---

## 控制台输出示例

### EdgeFilter
```
[EdgeFilter] Step 0: Removed frontend->emailservice | prior knowledge
[EdgeFilter] Step 0 summary: 4 edge(s) removed: frontend->emailservice, ...
[EdgeFilter] Run complete: 27 input edges -> 23 after removing 4 spurious edge(s).
```

### MetricClassifier
```
[MetricCLS] Initialized: 12 classes from 18 metrics
[MetricCLS] Deduplicated metric groups (each counted once):
  Class 2: ['rrt', 'rrt_max', 'rrt_p99']
[MetricCLS] Merged: 'rrt_p99' -> class 2 (with existing: ['rrt', 'rrt_max'])
```

---

## LLM 后端配置

编辑 `utils/llm_adapter.py`：

```python
PLATFORM = "mock"     # 不调用 LLM（测试默认）
PLATFORM = "apiyi"    # APIYI 中转（推荐）
PLATFORM = "openai"   # OpenAI 直连
PLATFORM = "volc"     # 火山引擎 Ark
```

---

## 文件结构

```
cllm_v8/
├── main.py
├── pipeline.py
├── datasets.py  /  evaluate.py  /  util.py
├── cfcbn/
│   ├── cbn_accumulator.py      CBN + 自适应 alpha tuner
│   ├── crfd_cbn_engine.py      CF-CBN 引擎
│   ├── rag_case_store.py       RAG 向量库（BoW 或 OpenAI 双后端）[新增]
│   └── alpha_strategies.py     RAG / Jaccard / Entropy 策略       [新增]
├── agents/
│   ├── collider_topology.py    EdgeFilter（Step-0 先验 + Step-2 LLM）
│   ├── llm_workflow.py         WF-A + WF-B + 全英文故障工单
│   ├── case_store.py           持久化 case 存档
│   ├── metric_classifier.py    在线指标去重（LLM）
│   └── knowledge/
│       ├── edge_prior_knowledge.json
│       └── collider_prior_knowledge.txt
├── ablation/
│   ├── ablation_run.py         A/B/C/D/CD 组   [新增 CD 组]
│   └── ablation_plot.py
├── data/
│   ├── ds25_faults.json  /  ds25_topo_s2s.json
│   └── tt_faults.json    /  tt_topo_s2s.json
├── utils/  (llm_adapter.py, anonymizer.py)
├── requirements.txt
├── README_EN.md
└── README_ZH.md    （本文件）
```

---

## v8 更新内容

| 序号 | 更新 | 涉及文件 |
|---|---|---|
| 1 | 故障工单全英文 + 每候选服务的原始异常指标（判断依据章节） | `agents/llm_workflow.py` |
| 2 | EdgeFilter / MetricClassifier 控制台详细输出 | `agents/collider_topology.py`, `agents/metric_classifier.py` |
| 3 | Group CD 联合消融（4 种配置，含 EdgeFilterAgent LLM 调用） | `ablation/ablation_run.py` |
| 4 | RAG 自适应 alpha（本地 BoW 默认 + OpenAI 后端可选） | `cfcbn/rag_case_store.py` |
| 5 | 三种 alpha 策略：RAG、Jaccard、Entropy | `cfcbn/alpha_strategies.py` |
| 6 | `--alpha-strategy` 命令行参数 | `main.py`, `pipeline.py` |
| 7 | 全源码零中文字符/符号 | 所有 `.py` 文件 |
