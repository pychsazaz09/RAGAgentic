from langchain_chroma import Chroma
from pathlib import Path
from typing import List,Tuple
from langchain_core.documents import Document


class ChromaStore:
    def __init__(self,embeddings) -> None:
        self.vectorstore=Chroma(
            collection_name="my_documents",
            embedding_function=embeddings,
            persist_directory="./db/chroma_db",
        )
    def add(self,documents:List[Document])->List[str]:
        ids=self.vectorstore.add_documents(documents)
        return ids
