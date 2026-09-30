from typing import List,Dict
from .chroma import ChromaStore
from .bm25 import BM25Store,seg
from .mineru_loader_file import MinerULoaderFile
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain.tools import tool
from ..dependency.embeddings import embeddings
from ..dependency.cross_encoder import cross_encoder_model


def reciprocal_rank_fusion(ranked_lists:List[List[Dict]],k:int=60):
    results_list={}
    scores_rrf={}
    for docs in results_list:
        for rank,doc in enumerate(docs,1):
            doc_id=doc["id"]
            scores_rrf[doc_id]=scores_rrf.get(doc_id,0)+1/(k+rank)
            results_list[doc_id]=doc
    sorted_rrf=sorted(scores_rrf.items(),key=lambda x:x[1],reverse=True)
    return [results_list[doc_id] for doc_id,score in sorted_rrf]


#之后可以换List[str]
def cross_encoder_rerank(query:str,docs:List[Dict],top_k:int=3):
    scores=cross_encoder_model.predict([(query,doc["content"])for doc in docs])
    for i,s in enumerate(scores):
        docs[i]["score"]=s
    sorted_docs=sorted(docs,key=lambda x:x["score"],reverse=True)
    sorted_docs= [doc for doc in sorted_docs if doc["score"] > 0]
    return sorted_docs[0:min(len(sorted_docs),top_k)]



class RAGKnowledge:
    def __init__(self,resources:str,top_k:int=3) -> None:
        self.top_k=top_k

        '''minerULoaderFile=MinerULoaderFile(resources)
        minerULoaderFile.load()'''

        headers_to_split_on=[
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        spliter=MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False,
        )
        #缺陷：目前还是没考虑多个输出文件
        with open("./resources/output/note.md","r",encoding="utf-8") as f:
            doc=f.read()
        chunks=spliter.split_text(doc)

        chroma_store=ChromaStore(embeddings=embeddings)
        chroma_store.add(chunks)
        self.chroma_retriever=chroma_store.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k},
        )

        bm25_store=BM25Store(chunks)
        self.bm_retriever=bm25_store.bm_retriever



    