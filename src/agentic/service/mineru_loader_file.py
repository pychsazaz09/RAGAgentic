from langchain_mineru import MinerULoader
from typing import List
from pathlib import Path
from langchain_core.documents import Document


class MinerULoaderFile:
    def __init__(self,dir:str) -> None:
        dir_resource=Path(dir)
        dir_resource.mkdir(parents=True,exist_ok=True)
        files=[
            str(p)
            for p in dir_resource.rglob("*")
            if p.suffix.lower() in [".pdf",".doc",".docx",".txt"]
        ]

        self.loader=MinerULoader(
            source=files,
            mode="flash",
            ocr=True
        )
    def load(self):
        docs=self.loader.load()

        output_dir=Path("./resources/output")
        output_dir.mkdir(parents=True,exist_ok=True)
        for doc in docs:
            file_name=Path(doc.metadata["source"]).stem
            file=output_dir/f"{file_name}.md"
            file.write_text(
                data=doc.page_content,
                encoding="utf-8",
            )
