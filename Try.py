#IMPORTANT LIB
import streamlit as st
import sentencepiece
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.chains.summarize import load_summarize_chain  
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel
import torch
import base64 
import importlib
import subprocess
import sys

#FILE LOADER AND PREPROCESSING
#tokenizer = AutoTokenizer.from_pretrained("MBZUAI/LaMini-Flan-T5-248M")
model="MBZUAI/LaMini-Flan-T5-248M"
summarizer = pipeline("summarization", model="MBZUAI/LaMini-Flan-T5-248M")

# If you also need tokenizer and model explicitly:
tokenizer = AutoTokenizer.from_pretrained(model)
model = AutoModelForSeq2SeqLM.from_pretrained(model) 

def file_preprocessing(file):   
    loader = PyPDFLoader(file)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    texts = text_splitter.split_documents(pages)
    final_text = ""
    for text in texts:
        print(text)
        final_text = final_text + text.page_content
    return final_text

#LLM PIPELINE
def llm_pipeline(filepath):
    pipe_sum = pipeline(
        'summarization',
        model = model,
        tokenizer = tokenizer,
        max_length = 500,
        min_length = 50,
    )
    input_text = file_preprocessing(filepath)
    result = pipe_sum(input_text)
    result = result[0]['summary_text']
    return result

#STREAMLIT CODE UI/UX
@st.cache_data
def display_PDF(file):
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    pdf_display = F'<ifram src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

st.set_page_config(layout='wide', page_title='Summerization App')

def main():
    st.title('Document Classification using LLM')
    uploaded_file = st.file_uploader("Upload your PDF", type=['pdf'])
    if uploaded_file is not None:
        if st.button("Summerize"):
            col1, col2 = st.columns(2)
            filepath = "Data_PDF/"+uploaded_file.name
            with open(filepath, 'wb') as temp_file:
                temp_file.write(uploaded_file.read())
            with col1:
                st.info("Uploaded PDF file")
                pdf_viewer = display_PDF(filepath)
            with col2:
                st.info("Summerization is below")    
                summary = llm_pipeline(filepath)
                st.success(summary)

if __name__ == '__main__':
    main()                            