import pandasai
import pandas as pd
import streamlit as st

from pandasai.llm.local_llm import LocalLLM
from pandasai import SmartDataframe

## ollama pull gemma:2b     ## ollama list

if "user_prompt" not in st.session_state:
    st.session_state.user_prompt = None

if "df" not in st.session_state:
    st.session_state.df = None

def instantiate_llm():
    model = LocalLLM(
        api_base='http://localhost:11434/v1/',
        model='gemma:2b'
    )
    return model

def update_prompt():
    print('Inside update_prompt ==> st.session_state.user_prompt - ', st.session_state.user_prompt)

def generate_response():
    print('st.session_state.prompt - ', st.session_state.user_prompt)
    if st.session_state.user_prompt :
        with st.spinner("Generating..."):
            response = st.session_state.df.generate(st.session_state.user_prompt)
            st.write(response)

            response2 = st.session_state.df.chat(st.session_state.user_prompt)
            st.write(response2)
            st.write('ANSWER ==>')
            pass


def main():
    model = instantiate_llm()

    st.title("Data Analysis with PandasAI")
    # st.header("Chat CSV with Pandasai")

    csv_file_uploaded = st.file_uploader(
        'Upload CSV file',
        type=['csv']
        )
    
    if csv_file_uploaded is not None:
        data = pd.read_csv(csv_file_uploaded)
        st.write(data.head(3))

        st.session_state.df = SmartDataframe(data, config={
            "llm": model,
            "custom_whitelisted_dependencies": ["pandasai"]
        })

        prompt = st.text_area("Enter your Prompt: ", on_change=update_prompt, key='user_prompt')
        st.write('What are the categories under the brand Lundberg')

        submit_btn =  st.button("Generate", on_click=generate_response)


if __name__ == '__main__':
    # main()

    model = instantiate_llm()

    st.title("Data Analysis with PandasAI")
    # st.header("Chat CSV with Pandasai")

    csv_file_uploaded = st.file_uploader(
        'Upload CSV file',
        type=['csv']
        )
    
    if csv_file_uploaded is not None:
        data = pd.read_csv(csv_file_uploaded)
        st.write(data.head(3))

        st.session_state.df = SmartDataframe(data, config={
            "llm": model
        })

        prompt = st.text_area("Enter your Prompt: ")
        st.write('What are the categories under the brand Lundberg')

        submit_btn =  st.button("Generate")

        if submit_btn :
            with st.spinner("Generating..."):
                response = st.session_state.df.chat(prompt)
                st.write(response)