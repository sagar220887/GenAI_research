from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.llms import HuggingFaceHub


from langchain.retrievers import BM25Retriever, EnsembleRetriever

import os

file_path = "../../data/yolo.pdf"
data_file = UnstructuredPDFLoader(file_path)
docs = data_file.load()
print(docs)