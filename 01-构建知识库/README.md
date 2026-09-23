# 第 1 节：构建知识库

## 30 秒理解

构建知识库不是“把文件丢进数据库”，而是把不同来源的数据统一成 `Document(page_content, metadata)`，按语义边界切成 Chunk，再用 Embedding 模型向量化并写入 Vector Store。最后把查询参数封装成 Retriever，供普通 RAG 或 Agent 调用。

```text
文件/网页/PDF -> Loader -> Document -> Splitter -> Chunk
                                       -> Embedding -> Vector Store -> Retriever
```

## 机制与数据流

### 1. 加载：统一数据形状

LangChain 的 Loader 屏蔽来源差异，统一返回 `list[Document]`：

```python
Document(
    page_content="真正参与检索的正文",
    metadata={"source": "来源", "page": 3}
)
```

常用 Loader：

- `TextLoader`：纯文本。
- `WebBaseLoader`：提取网页文本；动态页面、登录页和反爬页面可能失败。
- `CSVLoader`：通常一行一个 Document，可指定 `source_column`。
- `PyPDFLoader`：可按页或整篇加载；复杂表格、扫描件、跨页结构需更强解析器。

`metadata` 不是装饰品。引用来源、权限过滤、删除更新、评估定位都依赖它。

### 2. 切分：在完整语义与检索粒度之间折中

- `CharacterTextSplitter`：按分隔符/长度切，简单但容易破坏结构。
- `RecursiveCharacterTextSplitter`：依次尝试段落、换行、句号等分隔符，通用首选。
- `MarkdownHeaderTextSplitter`：按 Markdown 标题切，并把标题写进 metadata。

结构化文档可先按标题切，再对过长片段递归切。`chunk_overlap` 用少量重叠保留跨边界上下文，但过大会重复召回、增加成本。

### 3. 向量化：把语义投到同一空间

统一接口：

```python
embedding.embed_documents(["文档 A", "文档 B"])
embedding.embed_query("用户问题")
```

建库与查询必须使用同一模型、维度和预处理方式。换模型通常意味着重建索引。

### 4. 存储与检索

Vector Store 同时保存 Chunk、metadata、ID 与向量，并提供近似最近邻搜索：

```python
vectorstore.add_documents(chunks, ids=ids)
vectorstore.delete(ids=ids)
vectorstore.similarity_search(query, k=3)
vectorstore.similarity_search_with_relevance_scores(query, k=3)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)
docs = retriever.invoke(query)
```

Retriever 是统一的“取文档”接口；它不等于数据库，也不负责生成答案。

## 核心代码/API 速查

```python
from langchain_community.document_loaders import TextLoader, WebBaseLoader, PyPDFLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_chroma import Chroma
```

真实 Embedding 可按环境选择：

```python
from langchain_ollama import OllamaEmbeddings

embedding = OllamaEmbeddings(model="qwen3-embedding:0.6b")
```

或课程中的百炼适配器：

```python
from langchain_community.embeddings import DashScopeEmbeddings

embedding = DashScopeEmbeddings(
    model="text-embedding-v3",
    dashscope_api_key=os.environ["DASHSCOPE_API_KEY"],
)
```

密钥放环境变量，不写进源码。

## 最小可运行费曼 Demo

先安装：

```text
uv add langchain-core langchain-text-splitters
```

Demo 用一个可解释的 4 维 Embedding 和内存向量库，免 API Key；目标是看清接口与数据流，不代表生产检索质量。

```python
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter


class TinyEmbedding(Embeddings):
    words = ("退款", "物流", "会员", "发票")

    def _embed(self, text: str) -> list[float]:
        return [float(text.count(word)) for word in self.words]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


docs = [
    Document("退款需在签收后 7 天内申请。", metadata={"source": "售后.md"}),
    Document("物流单号可在订单详情查看。", metadata={"source": "物流.md"}),
    Document("会员积分在订单完成后到账。", metadata={"source": "会员.md"}),
]

splitter = RecursiveCharacterTextSplitter(chunk_size=30, chunk_overlap=5)
chunks = splitter.split_documents(docs)

store = InMemoryVectorStore(embedding=TinyEmbedding())
store.add_documents(chunks)
retriever = store.as_retriever(search_kwargs={"k": 1})

for doc in retriever.invoke("怎么申请退款？"):
    print(doc.page_content, doc.metadata)
```

你应该能指出：`split_documents` 保留 metadata；`add_documents` 内部调用 `embed_documents`；查询时 Retriever 调 `embed_query`，再返回最相近的 `Document`。

## 常见坑与版本提醒

- `Document.id` 与 `metadata["id"]` 用途不同；做增量更新时应维护稳定、可重复生成的业务 ID。
- 先 `delete(ids)` 再添加不是天然安全的全量更新方案；失败可能留下空窗，生产环境要考虑版本化索引或原子切换。
- 不要假定所有库的 score 方向一致。`similarity_search_with_score` 可能返回“距离”，`...with_relevance_scores` 才通常归一到相关性语义，仍需看适配器说明。
- `WebBaseLoader` 不等于浏览器，JS 渲染、登录态和页面噪声需额外处理，并遵守站点规则。
- PDF 文本顺序、表格、OCR 可能错误；Loader 成功只代表“读到了”，不代表“读对了”。
- Token 切分需要匹配实际模型 tokenizer；“字符数”等于“Token 数”是错误理解。
- Chroma 的持久化目录、collection 名、Embedding 维度应固定并纳入配置。

## 费曼复述

> Loader 像收件员，把各种文件装进统一信封 `Document`；Splitter 像剪辑师，把长文切成仍看得懂的小段；Embedding 像给每段标语义坐标；Vector Store 像按坐标摆放卡片；Retriever 则是规定每次拿回几张、按什么方式拿。

## 自测

1. 为什么 `metadata.source/page` 必须从加载阶段一直保留？
2. Chunk 太大和太小分别会导致什么问题？
3. `embed_documents` 与 `embed_query` 为什么必须处于同一向量空间？
4. `VectorStore` 与 `Retriever` 有什么区别？
5. 更换 Embedding 模型后，为什么通常需要重建索引？
6. 如何验证 PDF “加载成功”之外的解析质量？

