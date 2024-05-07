from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import CTransformers
# from src.helper import *

import streamlit as st
from io import StringIO
import os
import time
from PyPDF2 import PdfReader
from langchain.llms import CTransformers
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from langchain_community.document_loaders import UnstructuredURLLoader, MergedDataLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader, DirectoryLoader
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pickle
from langchain import HuggingFaceHub
from transformers import  AutoTokenizer
from ctransformers import AutoModelForCausalLM



# vector_store_file = "./model/saved/faiss_vector_store.pkl"
vector_db_directory = "./model/vectordb"
# llm_model_pkl_file = "./model/saved/llama-2-7b-chat.pkl"
# huggingface_embedding_pkl_file = "./model/saved/huggingface_embedding.pkl"

chain = None
vector_db = None

from dotenv import load_dotenv
load_dotenv()

HF_API_KEY = os.getenv('HUGGINGFACE_API_KEY')
HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')



if "processComplete" not in st.session_state:
    st.session_state.processComplete = None
if "new_data_source" not in st.session_state:
    st.session_state.new_data_source = False
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = None

if "data_chunks" not in st.session_state:
    st.session_state.data_chunks = None

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# file process button state
if "file_process" not in st.session_state:
    st.session_state.file_process = None

# Initialize chat history
if "upload_files" not in st.session_state:
    st.session_state.upload_files = None

st.markdown(
    """
<style>
    .st-emotion-cache-4oy321 {
        flex-direction: row-reverse;
        text-align: right;
    }
</style>
""",
    unsafe_allow_html=True,
)









#########################################################################################################

def load_data_source(loaded_files):
    for loaded_file in loaded_files:
        print('loaded_file - ', loaded_file)
        temp_file = create_temp_file(loaded_file)
        # temp_file = './tmp/74862151_1709607183894.pdf'

        # loader = PyPDFLoader(temp_file)
        loader = get_loader_by_file_extension(temp_file)
        print('loader - ', loader)
        data = loader.load()
        return data
    
def get_loader_by_file_extension(temp_file):
    file_split = os.path.splitext(temp_file)
    file_name = file_split[0]
    file_extension = file_split[1]
    print('file_extension - ', file_extension)

    
    if file_extension == '.pdf':
        loader = PyPDFLoader(temp_file)
        print('Loader Created for PDF file')
    
    elif file_extension == '.txt':
        loader = TextLoader(temp_file)

    elif file_extension == '.csv':
        loader = CSVLoader(temp_file)

    else :
        loader = UnstructuredFileLoader(temp_file)

    return loader

def create_temp_file(loaded_file):
    # save the file temporarily
    temp_file = f"./tmp/{loaded_file.name}"
    with open(temp_file, "wb") as file:
        file.write(loaded_file.getvalue())

    return temp_file



def get_data_chunks(data):
    recursive_char_text_splitter=RecursiveCharacterTextSplitter(
                                                chunk_size=500,
                                                chunk_overlap=50)
    documents=recursive_char_text_splitter.split_documents(data)
    # print('documents - ', documents)
    print('documents type - ', type(documents))
    print('documents length - ', len(documents))
    return documents


def save_model(model_path, model):
    # Save the FAISS index to a pickle file
    with open(model_path, "wb") as f:
        pickle.dump(model, f)


def load_model(model_path):
    if os.path.exists(model_path):
        with open(model_path, "rb") as f :
            model = pickle.load(f)
            return model
        


def create_embeddings():
    embeddings=HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2', 
            model_kwargs={'device':'cpu'}
    )
    return embeddings



def store_data_in_vectordb(documents, embeddings):
    try:
        current_vectordb = load_vectordb(vector_db_directory, embeddings)
        print('current_vectordb - ', current_vectordb)
    except:
        print('Exception inside storing data in vector db')

    new_knowledge_base =FAISS.from_documents(documents, embeddings)
    print('new_knowledge_base - ', new_knowledge_base)

    # Saving the new vector DB
    new_knowledge_base.save_local(vector_db_directory)
    return new_knowledge_base

    ## TODO
    # Adding new data to existing vector DB
#     updated_knowledge_base = new_knowledge_base.merge_from(current_vectordb)
#     print('updated_knowledge_base - ', updated_knowledge_base)

    # Saving the new vector DB
#     updated_knowledge_base.save_local(vector_db_directory)
#     return updated_knowledge_base




def load_vectordb(stored_directory, embeddings):
    loaded_vector_db = FAISS.load_local(stored_directory, embeddings)
    return loaded_vector_db
    

def get_llm_model():
    llm=CTransformers(
            model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
            model_type="llama",
            config={'max_new_tokens':128,
                    'temperature':0.01}
    )

    # llm = AutoModelForCausalLM.from_pretrained("./model/mistral-7b-instruct-v0.1.Q4_K_S.gguf", model_type="cpu")
    # llm = HuggingFaceHub(repo_id="mistralai/Mistral-7B-Instruct-v0.2", huggingfacehub_api_token=HF_API_KEY)
    print('LLM model Loaded')
    return llm



def get_prompt():
    template="""Use the following pieces of information to answer the user's question.
            If you dont know the answer just say you dont know, don't try to make up an answer.

            Context:{context}
            Question:{question}

            Only return the helpful answer below and nothing else
            Helpful answer
            """
    

    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    print('Prompt created')
    return prompt

def create_chain(llm, vector_store, prompt):
    chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type='stuff',
            retriever=vector_store.as_retriever(search_kwargs={'k': 2}),
            return_source_documents=False,
            chain_type_kwargs={'prompt': prompt}
    )
    print('Chain created')
    return chain

def create_conversational_chain(llm, vector_store, prompt):
    memory = ConversationBufferMemory()
    conversation_chain = ConversationalRetrievalChain(
        llm=llm,
        chain_type='stuff',
        retriever=vector_store.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt},
        memory=memory
    )
    print('Chain created')
    return conversation_chain

def get_similiar_docs(query,k=1,score=False):
  if score:
    similar_docs = vector_db.similarity_search_with_score(query,k=k)
  else:
    similar_docs = vector_db.similarity_search(query,k=k)
  return similar_docs


def process_for_new_data_source(uploaded_files):

        with st.spinner('Processing, Wait for it...'):
        
                # #Load the PDF File
                documents = load_data_source(uploaded_files)

                # #Split Text into Chunks
                st.session_state.data_chunks = get_data_chunks(documents)

                # #Load the Embedding Model
                embeddings = create_embeddings()

                # #Convert the Text Chunks into Embeddings and Create a FAISS Vector Store
                vector_db=store_data_in_vectordb(st.session_state.data_chunks, embeddings)
                
                llm = get_llm_model()

                qa_prompt = get_prompt()

                chain = create_chain(llm, vector_db, qa_prompt)

                st.text("Ready to go ...✅✅✅")
                st.session_state.processComplete = True

                return chain


def process_for_existing_source():

    st.session_state.conversation = True
    # #Load the Embedding Model
    embeddings = create_embeddings()
    vector_db = load_vectordb(vector_db_directory, embeddings)
    llm = get_llm_model()
    qa_prompt = get_prompt()
    chain = create_chain(llm, vector_db, qa_prompt)
    return chain


def get_response(user_query):
#     chain = process_for_new_data_source()
#     user_input = "Tell me about transformers in NLP"
#     user_input = "what is capital of india"
#     user_input = "What is the account no of Sagar"

#     similarity_search_value = vector_db.similarity_search(user_input)
#     print('similarity_search_value - ', similarity_search_value)

#     print('vector_db - ', vector_db)
    
    print('user_query - ', user_query)
    # print('Inside get_response  -', st.session_state.conversation)

    if st.session_state.conversation :

        result=st.session_state.conversation({'query':user_query}, return_only_outputs=True)
        print('result - ', result)
        ans = result['result']
        print(f"Answer:{ans}")
        return ans


def sidebar_design():
    with st.sidebar:
        st.title("Data sources")
        st.session_state.upload_files =  st.file_uploader(
            "Upload your file",
            type=['pdf','txt', 'csv', 'doc/docx'],
            accept_multiple_files=True
        )
        URL_TO_EXTRACT = st.text_input("Site URL")
        st.session_state.file_process = st.button("Process")



#######################################################################################################


# def main2():
#     st.subheader("Chat With Your Documents")
#     user_question = st.chat_input("Ask Question about your files.")


#     with st.sidebar:
#         st.title("Data sources")
#         uploaded_files =  st.file_uploader(
#             "Upload your file",
#             type=['pdf','txt', 'csv', 'doc/docx'],
#             accept_multiple_files=True
#         )
#         URL_TO_EXTRACT = st.text_input("Site URL")
#         process = st.button("Process")

#         if process:
#             if uploaded_files:
#                 st.session_state.new_data_source = True
#                 st.session_state.conversation = process_for_new_data_source(uploaded_files)
                

#             elif URL_TO_EXTRACT:
#                 st.session_state.new_data_source = True

#             else:
#                 st.session_state.new_data_source = False

#         else:
#             st.session_state.processComplete = True

#     if st.session_state.processComplete == True :



#         with st.chat_message("assistant"):
#             st.write("Hello Human")

#             if user_question :
#                     # user_input = "What is the account no of Sagar"
#                     st.text(f'You : {user_question}', )
#                     response = get_response(user_question)
#                     st.text(f'Bot : {response}', )



def main():
    st.title("Chat with Your documents !!")

    with st.sidebar:
        st.title("Data sources")
        st.session_state.upload_files =  st.file_uploader(
            "Upload your file",
            type=['pdf','txt', 'csv', 'doc/docx'],
            accept_multiple_files=True
        )
        URL_TO_EXTRACT = st.text_input("Site URL")
        st.session_state.file_process = st.button("Process")

        if st.session_state.file_process:
            if st.session_state.upload_files:
                st.session_state.new_data_source = True
                st.session_state.conversation = process_for_new_data_source(st.session_state.upload_files)

            elif URL_TO_EXTRACT:
                st.session_state.new_data_source = True





    with st.chat_message("assistant"):
        st.write("Hello Human, How do I help U.")

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        print('message - ', message)
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Ask Question about your files."):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        print('st.session_state.new_data_source ---> ', st.session_state.new_data_source)
        if st.session_state.new_data_source == False:
            process_for_existing_source()

        
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            print('Inside st.chat_message("assistant")')
            with st.spinner('Processing ...'):
                response = get_response(prompt)
                st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        


if __name__ == '__main__':
    main()