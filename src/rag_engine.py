
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_groq import ChatGroq

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS

load_dotenv()


# --------------------------------------------------
# LLM
# --------------------------------------------------

def get_llm(
    model: str = "llama-3.1-8b-instant",
    temperature: float = 0.2,
):
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. Add it in Streamlit Secrets."
        )

    return ChatGroq(
        model=model,
        temperature=temperature,
    )


# --------------------------------------------------
# Embeddings
# --------------------------------------------------

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# --------------------------------------------------
# Stage 1: Load Documents
# --------------------------------------------------

def load_text_file(
    file_path: str,
    source_name: str,
    doc_type: str,
) -> List[Document]:

    path = Path(file_path)

    text = path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    return [
        Document(
            page_content=text,
            metadata={
                "source": source_name,
                "doc_type": doc_type,
            },
        )
    ]


def create_documents(
    resume_text: str,
    jd_text: str,
) -> List[Document]:

    return [
        Document(
            page_content=resume_text,
            metadata={
                "source": "uploaded_resume",
                "doc_type": "resume",
            },
        ),
        Document(
            page_content=jd_text,
            metadata={
                "source": "uploaded_job_description",
                "doc_type": "job_description",
            },
        ),
    ]


# --------------------------------------------------
# Stage 2: Chunking
# --------------------------------------------------

def split_documents(
    docs: List[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> List[Document]:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            ".",
            " ",
            "",
        ],
    )

    return splitter.split_documents(docs)


# --------------------------------------------------
# Stage 3 + 4: Embeddings + FAISS
# --------------------------------------------------

def build_vectorstore(
    chunks: List[Document],
):
    embeddings = get_embeddings()

    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    return vectorstore


# --------------------------------------------------
# Stage 5: Retrieval
# --------------------------------------------------

def retrieve_context(
    vectorstore,
    query: str,
    k: int = 5,
) -> Tuple[str, List[Document]]:

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": k}
    )

    docs = retriever.invoke(query)

    context = "\n\n".join(
        [
            f"SOURCE: {d.metadata}\nCONTENT:\n{d.page_content}"
            for d in docs
        ]
    )

    return context, docs


# --------------------------------------------------
# Stage 6: Generation
# --------------------------------------------------

def run_career_coach(
    vectorstore,
    resume_text: str,
    jd_text: str,
    question: str,
):

    llm = get_llm()

    retrieval_query = f"""
    Resume content and job description content relevant to this question:

    {question}
    """

    context, source_docs = retrieve_context(
        vectorstore,
        retrieval_query,
        k=6,
    )

    prompt = ChatPromptTemplate.from_template(
        """
You are an expert Career Coach.

Use ONLY the retrieved context.

CONTEXT:
{context}

QUESTION:
{question}

Provide:

1. Current Match Summary
2. Strengths
3. Missing Skills
4. Resume Improvements
5. Suggested Projects
6. Interview Preparation Tips

Keep the answer practical, beginner-friendly and actionable.
"""
    )

    chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    answer = chain.invoke(
        {
            "context": context,
            "question": question,
        }
    )

    return answer, source_docs


# --------------------------------------------------
# Complete Report
# --------------------------------------------------

def generate_complete_report(
    vectorstore,
    resume_text: str,
    jd_text: str,
):

    question = """
Analyze the resume against the job description.

Provide:

- ATS Score
- Skill Match
- Missing Skills
- Resume Improvements
- Project Suggestions
- Interview Questions
- Final Action Plan
"""

    return run_career_coach(
        vectorstore,
        resume_text,
        jd_text,
        question,
    )
