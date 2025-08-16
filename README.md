
# 📄Document Classification for QLink

This project implements a document processing and summarization pipeline using Streamlit, Hugging Face Transformers, and LangChain.
It enables users to upload PDF documents, preprocess them, and generate concise AI-powered summaries using LaMini-Flan-T5 models.


## Features

📂 PDF Upload & Processing – Load single or multiple PDF files.

✂️ Text Splitting – Breaks large documents into smaller chunks for efficient processing.

🤖 Summarization – Uses MBZUAI/LaMini-Flan-T5-248M to generate accurate summaries.

⚡ Streamlit UI – Simple and interactive web app interface.

🧩 LangChain Integration – Efficient document handling with RecursiveCharacterTextSplitter.


## Installation

Install my-project with npm

```bash
git clone https://github.com/Shaily4504/Smart-Document-Classifier-for-QLink.git
cd Smart-Document-Classifier-for-QLink

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows

# Install requirements
pip install -r requirements.txt
```
    
## Usage/Examples

```python
streamlit run Try.py
```


## Model Used
MBZUAI/LaMini-Flan-T5-248M

Fine-tuned version of T5 optimized for summarization tasks.
Lightweight and efficient for document classification pipelines.
## License

This project is licensed under the MIT License – feel free to use and modify.

[MIT](https://choosealicense.com/licenses/mit/)


