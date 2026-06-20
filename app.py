from __future__ import annotations

import streamlit as st

from src.file_utils import read_uploaded_file
from src.rag_engine import (
    create_documents,
    split_documents,
    build_vectorstore,
    run_career_coach,
    generate_complete_report,
)

st.set_page_config(
    page_title="RAG Career Coach ",
    page_icon="🎯",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main-title {font-size: 42px; font-weight: 800; margin-bottom: 0px; color: #4F46E5;}
    .subtitle {font-size: 18px; color: #6B7280; margin-bottom: 25px;}
    .stage-box {padding: 14px; border-radius: 12px; background: #F5F3FF; border: 1px solid #DDD6FE; margin-bottom: 10px;}
    .success-box {padding: 12px; border-radius: 10px; background: #ECFEFF; border: 1px solid #67E8F9;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title"> RAG Career Coach - Traditional RAG System </div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Resume + Job Description → RAG Pipeline → Skill Gap Analysis, Resume Suggestions & Interview Prep</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("⚙️ RAG Settings")
    chunk_size = st.slider("Chunk size", 300, 1500, 800, 100)
    chunk_overlap = st.slider("Chunk overlap", 0, 400, 150, 50)
    st.divider()
    st.markdown("### RAG Stages")
    st.markdown("1. Load Resume & JD")
    st.markdown("2. Split into Chunks")
    st.markdown("3. Convert to Embeddings")
    st.markdown("4. Store in FAISS Vector Store")
    st.markdown("5. Retrieve Relevant Context")
    st.markdown("6. Generate Career Advice")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📄 Upload Resume")
    resume_file = st.file_uploader("Upload resume (.txt, .pdf, .docx)", type=["txt", "pdf", "docx"], key="resume")
    resume_text_input = st.text_area("Or paste resume text", height=220, key="resume_text")

with col2:
    st.subheader("💼 Upload Job Description")
    jd_file = st.file_uploader("Upload JD (.txt, .pdf, .docx)", type=["txt", "pdf", "docx"], key="jd")
    jd_text_input = st.text_area("Or paste job description", height=220, key="jd_text")

resume_text = read_uploaded_file(resume_file) if resume_file else resume_text_input
jd_text = read_uploaded_file(jd_file) if jd_file else jd_text_input

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chunks" not in st.session_state:
    st.session_state.chunks = []

st.divider()

if st.button("🚀 Build Career Coach RAG Index", type="primary"):
    if not resume_text.strip() or not jd_text.strip():
        st.error("Please upload or paste both Resume and Job Description.")
    else:
        with st.spinner("Running RAG stages: loading → chunking → embeddings → vector database..."):
            docs = create_documents(resume_text, jd_text)
            chunks = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            vectorstore = build_vectorstore(chunks)

            st.session_state.vectorstore = vectorstore
            st.session_state.chunks = chunks

        st.success("RAG index created successfully!")
        c1, c2, c3 = st.columns(3)
        c1.metric("Documents", "2")
        c2.metric("Chunks", len(st.session_state.chunks))
        c3.metric("Vector DB", "ChromaDB")

if st.session_state.vectorstore:
    st.markdown("### ✅ Ask Career Questions")

    quick_questions = [
        "How well does this resume match the job description?",
        "What are the missing skills for this role?",
        "How can I improve this resume?",
        "Suggest 3 projects to become suitable for this role.",
        "Generate interview questions based on the skill gaps.",
    ]

    selected = st.selectbox("Choose a question", quick_questions)
    custom_question = st.text_input("Or ask your own question")
    final_question = custom_question if custom_question else selected

    if st.button("🤖 Get Career Coach Answer"):
        with st.spinner("Retrieving context and generating answer..."):
            answer, sources = run_career_coach(
                st.session_state.vectorstore,
                resume_text,
                jd_text,
                final_question,
            )

        st.markdown("## 🎯 Career Coach Response")
        st.write(answer)

        with st.expander("🔍 Retrieved Context Used by RAG"):
            for i, doc in enumerate(sources, 1):
                st.markdown(f"#### Source Chunk {i}")
                st.caption(str(doc.metadata))
                st.write(doc.page_content[:1200])

    st.divider()

    if st.button("📊 Generate Complete Career Report"):
        with st.spinner("Generating complete RAG-based career report..."):
            report, sources = generate_complete_report(
                st.session_state.vectorstore,
                resume_text,
                jd_text,
            )
        st.markdown("## 📊 Complete Career Report")
        st.write(report)

else:
    st.info("Upload/paste Resume and Job Description, then click 'Build Career Coach RAG Index'.")

st.divider()
st.markdown("### 🧠 How this Traditional RAG Project Works")
st.markdown(
    """
### Traditional RAG Workflow

1. **Document Loading**
   - Resume and Job Description are converted into LangChain Documents.

2. **Chunking**
   - Large documents are split into smaller overlapping chunks using RecursiveCharacterTextSplitter.

3. **Embedding Generation**
   - Each chunk is converted into a dense vector using HuggingFace Embeddings.

4. **Vector Storage**
   - Embeddings are stored in a FAISS Vector Database for efficient retrieval.

5. **Query Processing**
   - User questions are converted into embeddings.

6. **Semantic Retrieval**
   - FAISS retrieves the most relevant chunks from the Resume and Job Description.

7. **LLM-Powered Analysis**
   - Groq Llama 3.1 uses the retrieved context to generate skill-gap analysis, resume suggestions, project recommendations, and interview preparation guidance.
"""
)
