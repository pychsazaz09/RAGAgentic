import pkuseg
import bm25s
from pathlib import Path
from typing import List
from langchain_core.documents import Document

seg=pkuseg.pkuseg()

class BM25Store:
    def __init__(self,documents:List[Document],index_dir:str="./db/bm25_index",k1:float=1.5,b:float=0.75) -> None:
        dir=Path(index_dir)
        dir.mkdir(parents=True,exist_ok=True)
        file=dir/"data.csc.index.npy"

        metadata_corpus=[
            {
                "id":doc.metadata.get("source",str(i)),
                "content":doc.page_content,
            }
            for i,doc in enumerate(documents,1)
        ]

        if file.exists():
            self.bm_retriever=bm25s.BM25.load(
                str(dir),
                load_corpus=True,
            )
        else:
            self.bm_retriever=bm25s.BM25(
                k1=k1,
                b=b,
                corpus=metadata_corpus
            )
            corpus_fast=[seg.cut(doc["content"])for doc in metadata_corpus]
            self.bm_retriever.index(corpus=corpus_fast)

            self.bm_retriever.save(str(dir),metadata_corpus)
            self.metadete_corpus=metadata_corpus