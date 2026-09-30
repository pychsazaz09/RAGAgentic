
import time
from langchain.tools import tool
from .bm25 import seg
from .rag_knowledge import RAGKnowledge,reciprocal_rank_fusion,cross_encoder_rerank


rag_knowledge=RAGKnowledge(resources="./resources")

@tool
def search(query:str)->str:
    '''
    搜索知识库，获取关键知识，需要查找资料时调用
    '''
    start=time.perf_counter()
    docs_chroma=rag_knowledge.chroma_retriever.invoke(query)
    end=time.perf_counter()
    print(f"向量检索耗时{end-start}")

    start=time.perf_counter()
    query_tokens=[seg.cut(query)]
    docs_bm=rag_knowledge.bm_retriever.retrieve(query_tokens=query_tokens,k=rag_knowledge.top_k)
    end=time.perf_counter()
    print(f"精确检索耗时{end-start}")

    vec_docs=[
        {
            "id":doc.id,
            "content":doc.page_content
        }
        for doc in docs_chroma
    ]
    acc_docs=[
        {
            "id":doc["id"],
            "content":doc["content"]
        }
        for doc in docs_bm
    ]
    rrf_docs=reciprocal_rank_fusion([vec_docs,acc_docs])
    start=time.perf_counter()
    final_docs=cross_encoder_rerank(query=query,docs=rrf_docs)
    end=time.perf_counter()
    print(f"重排序耗时:{end-start}")

    docs_content="\n\n".join(doc["content"] for doc in final_docs)
    print(f"检索问题:{query}")
    print(f"相关文档:{docs_content}")

    return docs_content
