import streamlit as st

if "uploader_visible" not in st.session_state:
    st.session_state["uploader_visible"] = False

def show_upload_component(component_state):
    if component_state:
        file = st.file_uploader(
            "Upload your file",
            type=['pdf','txt', 'csv', 'doc/docx'],
            accept_multiple_files=True
        )

def toggle_uploader(btn_state:bool):
    st.session_state["uploader_visible"] = not btn_state
    if st.session_state["uploader_visible"]:
        show_upload_component()
    


with st.sidebar:
    action_btn = st.button(
        label="Upload files or urls",
        key='user_action_btn',
        on_click=toggle_uploader,
        args=(st.session_state["uploader_visible"],)
    )
    