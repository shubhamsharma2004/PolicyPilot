from langchain_community.document_loaders import PyPDFLoader
import os

path = r"C:\company\policies\Policies.pdf"
print("Exists:", os.path.exists(path))
if os.path.exists(path):
    docs = PyPDFLoader(path).load()
    print("Pages loaded:", len(docs))
    if docs:
        print("First 200 chars:", docs[0].page_content[:200])