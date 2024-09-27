import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter



def create_embeddings():
    embeddings=HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2', 
            model_kwargs={'device':'cpu'}
    )
    return embeddings

def get_data_chunks(data):
    recursive_char_text_splitter=RecursiveCharacterTextSplitter(
                                                chunk_size=500,
                                                chunk_overlap=50)
    documents=recursive_char_text_splitter.split_documents(data)
    # print('documents - ', documents)
    print('documents type - ', type(documents))
    print('documents length - ', len(documents))
    return documents

current_path = os.getcwd()
parent_path = Path(current_path).parent.parent

# Load PDFs
temp_file_1 = os.path.join(parent_path, 'data', 'Gen AI.pdf')
loader_1 = PyPDFLoader(temp_file_1)
data_1 = loader_1.load()

temp_file_2 = os.path.join(parent_path, 'data', 'MachineTranslationwithAttention.pdf')
loader_2 = PyPDFLoader(temp_file_2)
data_2 = loader_2.load()

documents = data_1 + data_2

documents = get_data_chunks(documents)

embeddings = create_embeddings()
vector_data = [ embeddings.embed_query(doc.page_content) for doc in documents]
print(vector_data[1])