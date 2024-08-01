import streamlit as st 
from langchain_community.llms import Ollama 
import pandas as pd
from pandasai import SmartDataframe

from pandasai.connectors import PostgreSQLConnector
from pandasai import SmartDataframe
from pandasai.llm.local_llm import LocalLLM

from langchain import HuggingFaceHub
import os
from dotenv import load_dotenv
load_dotenv()


HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')
os.environ['HUGGINGFACEHUB_API_TOKEN'] = HUGGINGFACEHUB_API_TOKEN

# llm = Ollama(model="mixtral")


llm = HuggingFaceHub(repo_id="mistralai/Mistral-7B-Instruct-v0.2", huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN)

st.title("Data Analysis with PandasAI")

uploader_file = st.file_uploader("Upload a CSV file", type= ["csv"])

if uploader_file is not None:
    data = pd.read_csv(uploader_file)
    st.write(data.head(3))
    df = SmartDataframe(data, config={"llm": llm, "custom_whitelisted_dependencies": ["pandasai"]})
    prompt = st.text_area("Enter your prompt:")

    if st.button("Generate"):
        if prompt:
            with st.spinner("Generating response..."):
                st.write(df.chat(prompt))
        else:
            st.warning("Please enter a prompt!")