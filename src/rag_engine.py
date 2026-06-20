from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

DB_DIR = "career_coach_chroma_db"


def get_llm(model: str = "llama-3.1-8b-instant", temperature: float = 0.2):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. Create a .env file in your project root and add: GROQ_API_KEY=your_key_here"
        )
    return ChatGroq(model=model, temperature=temperature)


def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# -----------------------------
# Stage 1: Load Documents
# -----------------------------
def load_text_file(file_path: str, source_name: str, doc_type: str) -> List[Document]:
    path = Path(file_path)
    text = path.read_text(encoding="utf-8", errors="ignore")
    return [Document(page_content=text, metadata={"source": source_name, "doc_type": doc_type})]


def create_documents(resume_text: str, jd_text: str) -> List[Document]:
    """Wrap the resume and job description into LangChain Document objects, ready for chunking."""
    return [
        Document(page_content=resume_text, metadata={"source": "uploaded_resume", "doc_type": "resume"}),
        Document(page_content=jd_text, metadata={"source": "uploaded_job_description", "doc_type": "job_description"}),
    ]


# -----------------------------
# Stage 2: Split Documents
# -----------------------------
def split_documents(docs: List[Document], chunk_size: int = 800, chunk_overlap: int = 150) -> List[Document]:
    """Break large resume/JD text into overlapping chunks for better retrieval granularity."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    return splitter.split_documents(docs)


# -----------------------------
# Stage 3 + 4: Embeddings + Vector DB
# -----------------------------
def build_vectorstore(chunks: List[Document], persist_directory: str = DB_DIR):
    """Embed every chunk and persist it into a fresh ChromaDB collection."""
    if Path(persist_directory).exists():
        shutil.rmtree(persist_directory)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=persist_directory,
        collection_name="career_coach_rag",
    )
    return vectorstore


# -----------------------------
# Stage 5: Retrieve Context
# -----------------------------
def retrieve_context(vectorstore, query: str, k: int = 5) -> Tuple[str, List[Document]]:
    """Run a similarity search and stitch the top-k chunks into a single context block."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(query)
    context = "\n\n".join([f"SOURCE: {d.metadata}\nCONTENT:\n{d.page_content}" for d in docs])
    return context, docs


# -----------------------------
# Stage 6: Generate Answer
# -----------------------------
def run_career_coach(vectorstore, resume_text: str, jd_text: str, question: str):
    """Retrieve relevant context, then let the LLM answer the user's career question grounded in it."""
    llm = get_llm()

    retrieval_query = f"""
    Resume content and job description content relevant to this career coaching question:
    {question}
    """

    context, source_docs = retrieve_context(vectorstore, retrieval_query, k=6)

    prompt = ChatPromptTemplate.from_template("""
You are an expert AI Career Coach for students, freshers and working professionals.
Use ONLY the given context from the resume and job description.
Do not invent skills, experience or job requirements.

CONTEXT:
{context}

USER QUESTION:
{question}

Give a clear, practical answer with these sections when relevant:
1. Current Match Summary
2. Strengths
3. Missing Skills / Gaps
4. Recommended Improvements
5. Suggested Projects
6. Interview Preparation Tips

Keep the answer simple, actionable and beginner-friendly.
""")

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    return answer, source_docs


def generate_complete_report(vectorstore, resume_text: str, jd_text: str):
    """Produce a single, comprehensive career report covering match score, gaps, and prep tips."""
    question = """
    Analyze this resume against this job description. Provide ATS-style score, skill match, missing skills,
    resume improvement suggestions, project suggestions, and interview questions.
    """
    return run_career_coach(vectorstore, resume_text, jd_text, question)