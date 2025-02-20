import os
import json
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

persist_directory = r'./chroma.db'
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding_function
)

def store_file(file_path):   
    with open(file_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
        
    content_str = json.dumps(json_data["content"], ensure_ascii=False)
    doc = Document(page_content=content_str,metadata = json_data["metadata"])
    print(file_path)
    print(content_str,json_data["metadata"])
    print('-'*50)
    vectorstore.add_documents([doc])
    
def store_directory(directory_path):
    print('initialized')
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.json'):
                store_file(os.path.join(root, file))
                print(file)
                
store_directory(r"processed_data")

print('All files gone through')