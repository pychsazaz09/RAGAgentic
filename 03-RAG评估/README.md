# 第 3 节：RAG 评估

## 30 秒理解

RAG 的最终回答坏了，可能是“没搜到”、也可能是“搜到了但模型乱说”。因此必须分开评估：

```text
检索阶段：搜得准不准、全不全、排序好不好
生成阶段：是否忠于资料、是否切题、是否被噪声带偏
```

评估样本至少保留四项：用户问题、实际召回上下文、实际回答、参考答案。只有它们来自同一次真实运行，分数才可诊断。

## 机制与数据流

```text
测试问题 user_input
      │
      ├─ RAG 实际检索 -> retrieved_contexts ─┐
      └─ RAG 实际回答 -> response             ├─ Judge -> 指标
人工/高质量参考答案 -> reference ─────────────┘
```

新版 RAGAS 资料中的字段对照：

| 当前字段 | 旧教程常见字段 | 含义 |
|---|---|---|
| `user_input` | `question` | 用户问题 |
| `retrieved_contexts` | `contexts` | 实际召回的正文列表 |
| `response` | `answer` | RAG 实际回答 |
| `reference` | `ground_truth` | 参考答案 |

不要在评估时重新检索一次来补 `retrieved_contexts`，否则评估的不是生成回答时真正看到的上下文。应在一次运行中记录检索结果、模型回答、配置、耗时与版本。

## 六个核心指标

| 指标 | 回答的问题 | 方向 | 主要依赖 |
|---|---|---:|---|
| Context Precision | 召回结果里相关内容多吗，相关项是否靠前？ | 越高越好 | 问题、参考答案、上下文 |
| Context Recall | 参考答案需要的事实召回全了吗？ | 越高越好 | 问题、参考答案、上下文 |
| Context Entity Recall | 人名、术语、数字等实体覆盖全吗？ | 越高越好 | 参考答案、上下文 |
| Faithfulness | 回答中的陈述都能由上下文支持吗？ | 越高越好 | 问题、回答、上下文 |
| Answer/Response Relevancy | 回答是否切题？ | 越高越好 | 问题、回答、Embedding |
| Noise Sensitivity | 模型是否被无关上下文带偏？ | 越低越好 | 问题、回答、参考、上下文 |

诊断顺序：

- Recall 低：先查 Loader、Chunk、Embedding、Top-K、查询改写和召回策略。
- Precision 低：减少噪声、调阈值、融合与重排。
- Recall/Precision 高但 Faithfulness 低：生成 Prompt、模型约束或引用机制有问题。
- Faithfulness 高但 Relevancy 低：回答虽然有依据，但没有正面回答问题。
- Noise Sensitivity 高：上下文含干扰项，模型筛选证据能力不足。

课程给出的阈值只能当示例目标，不是跨业务标准。应按风险、数据集难度与人工基线制定门槛。

## 核心代码/API

课程资料采用 RAGAS 0.4.x 风格：

```python
from ragas import Dataset, experiment
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextEntityRecall,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
    NoiseSensitivity,
)
```

单个指标的调用形状：

```python
metric = Faithfulness(llm=evaluator_llm)
result = await metric.ascore(
    user_input=row["user_input"],
    response=row["response"],
    retrieved_contexts=row["retrieved_contexts"],
)
print(result.value)
```

批量、可追溯实验：

```python
from pydantic import BaseModel
from ragas import experiment

class EvalResult(BaseModel):
    context_recall: float
    faithfulness: float

@experiment(experiment_model=EvalResult)
async def run_rag_eval(row):
    recall = await ContextRecall(llm=evaluator_llm).ascore(
        user_input=row["user_input"],
        reference=row["reference"],
        retrieved_contexts=row["retrieved_contexts"],
    )
    faith = await Faithfulness(llm=evaluator_llm).ascore(
        user_input=row["user_input"],
        response=row["response"],
        retrieved_contexts=row["retrieved_contexts"],
    )
    return EvalResult(
        context_recall=recall.value,
        faithfulness=faith.value,
    )
```

## 最小可运行费曼 Demo

先固定版本并安装。下面沿用课程的 RAGAS 0.4.x API 形状；实际项目应在锁文件中记录经过验证的精确版本组合。

```text
uv add "ragas>=0.4,<0.5" openai
```

```python
import asyncio
import os
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness


async def main():
    client = AsyncOpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
    )
    judge = llm_factory(model="deepseek-chat", client=client)

    metric = Faithfulness(llm=judge)
    result = await metric.ascore(
        user_input="退款期限是多久？",
        retrieved_contexts=["商品签收后 7 天内可申请退款。"],
        response="商品签收后 7 天内可以申请退款。",
    )
    print("faithfulness =", result.value)


asyncio.run(main())
```

亲手做两个改动验证理解：

1. 把回答改成“30 天内退款”，Faithfulness 应下降。
2. 把回答改成只讨论物流，即使没有编事实，也应再用 Relevancy 指标检查是否跑题。

## 建立可信评估集

1. 覆盖真实用户问题：直接问、口语问、错别字、专有词、复合问题、无法回答的问题。
2. `reference` 由领域专家审核，避免 Judge 在错误标准答案上“正确打分”。
3. 保存真正使用过的 `retrieved_contexts`，以及 Chunk ID、来源和排名。
4. 划分开发集与回归集；不要一边调参数一边反复在同一小集合上宣称提升。
5. 先做小样本冒烟测试，再批量跑，控制费用、并发、超时与限流。
6. LLM-as-a-Judge 分数要用人工抽检校准；高风险业务不能只看自动指标。

## 常见坑与版本提醒

- RAGAS 0.4.x 与旧版在 Dataset、字段名、指标导入和批量 API 上差异很大；复制示例前先检查已安装版本。
- 不建议通过手工修改 `.venv/site-packages` 或创建空壳模块修复依赖；这种做法不可复现。应优先升级/降级到兼容组合、锁版本并记录上游问题。
- Judge 模型不是客观真理；同一模型既生成又评分会产生偏好，应对关键样本做人工复核或多 Judge 对照。
- 随机抽 10% 只适合快速冒烟，不代表统计上可靠；固定随机种子并保证类别覆盖。
- 全局字典记录上下文在线程/并发下可能串数据。生产评估应按 run ID 保存工具输出，避免用问题文本当唯一键。
- `list(set(contexts))` 会破坏顺序，而 Context Precision 关心排名；评估数据必须保留原始顺序。
- 评估样本不能泄漏到 Prompt 或模型上下文；参考答案只给 Judge，不给被测 RAG。
- 分数提升要同时查看质量、延迟与成本，避免用昂贵重排换来不可接受的响应时间。

## 费曼复述

> 评估 RAG 像检查快递。先看仓库有没有捡对货、漏没漏货、重要货是否排在前面；再看配送员有没有把货送错、添油加醋或被杂物干扰。四份凭证——问题、实际检索、实际回答、标准答案——必须来自同一次测试，才能追责到具体环节。

## 自测

1. Context Recall 低而 Faithfulness 高，系统可能发生了什么？
2. 为什么评估时不能重新跑一次 Retriever 来生成 contexts？
3. `list(set(contexts))` 为什么会破坏 Context Precision 的可信度？
4. Faithfulness 与 Answer Relevancy 分别能发现什么不同问题？
5. 为什么 LLM-as-a-Judge 仍需要人工抽检？
6. 一次 RAG 优化上线前，除平均分外还应记录哪些信息？

