# 🎯 RAG Career Coach

### Intelligent Resume Analysis, Skill Gap Detection & Career Guidance using Traditional RAG

🚀 **Live Demo:**

https://rag-career-coach.onrender.com

---

### Features

* Resume vs Job Description Matching
* Skill Gap Analysis
* Resume Improvement Suggestions
* Project Recommendations
* Interview Question Generation
* ATS-Oriented Career Report

### Tech Stack

* Streamlit
* LangChain
* HuggingFace Embeddings
* FAISS Vector Store
* Groq Llama 3.1
* Traditional RAG Pipeline

### Live Demo

Try the application here:

[Launch RAG Career Coach](https://rag-career-coach-eszo44jpfb7upecpeksbqc.streamlit.app/?utm_source=chatgpt.com)

[![Live Demo](https://img.shields.io/badge/Live-Demo-success)](https://rag-career-coach-eszo44jpfb7upecpeksbqc.streamlit.app/)
<img width="1919" height="944" alt="image" src="https://github.com/user-attachments/assets/34aa1090-b125-4a16-b485-d273cec1ba7b" />
<img width="1919" height="946" alt="image" src="https://github.com/user-attachments/assets/7221d40b-7263-41e4-8eaa-399e64d56d38" />


## 🏗️ Architecture

Resume + Job Description
        ↓
Document Loading
        ↓
Text Chunking
        ↓
HuggingFace Embeddings
        ↓
FAISS Vector Store
        ↓
Similarity Retrieval
        ↓
Groq Llama 3.1
        ↓
Career Guidance Report

## 🔄 RAG Workflow

1. Load Resume and Job Description
2. Split documents into chunks
3. Generate embeddings using HuggingFace
4. Store embeddings in FAISS
5. Retrieve relevant chunks using similarity search
6. Pass retrieved context to Groq Llama 3.1
7. Generate career guidance and recommendations

## ⚙️ Installation

```bash
git clone https://github.com/Pavithrareddy2702/RAG-Career-Coach.git

cd RAG-Career-Coach

pip install -r requirements.txt

streamlit run app.py


---

```md
## 📂 Project Structure

RAG-Career-Coach/
│
├── app.py
├── requirements.txt
├── README.md
│
├── src/
│   ├── rag_engine.py
│   └── file_utils.py
│
└── data/


## 🚀 Future Enhancements


- Cover Letter Generator
- LinkedIn Profile Analyzer
- Multi-Resume Comparison
- Job Recommendation Engine






