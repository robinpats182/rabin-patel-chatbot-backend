import os
import json
import chromadb
from pypdf import PdfReader
from google import genai
from google.genai import types

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY is not set — check your .env file or Render environment variables")

client_ai = genai.Client(api_key=api_key)

chroma_client = chromadb.EphemeralClient()
collection = chroma_client.get_or_create_collection("portfolio")

SYSTEM_PROMPT = """You are a helpful assistant answering questions about Rabin Patel \
for recruiters and visitors to his portfolio site. Only answer using the provided \
context. If the context doesn't contain the answer, say you don't have that \
information and suggest they reach out directly via email. Keep answers concise \
and professional. Speak about Rabin in the third person."""

def embed_texts(texts: list[str], task_type: str) -> list[list[float]]:
    result = client_ai.models.embed_content(
        model="text-embedding-004",
        contents=texts,
        config=types.EmbedContentConfig(task_type=task_type),
    )
    return [e.values for e in result.embeddings]

def chunk_text(text: str, size: int = 500, overlap: int = 50) -> list[str]:
    words = text.split()
    return [" ".join(words[i : i + size]) for i in range(0, len(words), size - overlap)]

def build_index():
    docs, ids, metadatas = [], [], []

    with open("data/profile.json") as f:
        profile = json.load(f)

    for i, job in enumerate(profile["experience"]):
        text = f"{job['role']} at {job['company']} ({job['dates']}). {job['summary']} " + " ".join(job["bullets"])
        docs.append(text); ids.append(f"exp-{i}"); metadatas.append({"source": "experience"})

    for i, proj in enumerate(profile["projects"]):
        text = f"Project: {proj['title']}. {proj['problem']} {proj['description']} Result: {proj['result']}"
        docs.append(text); ids.append(f"proj-{i}"); metadatas.append({"source": "project"})

    docs.append(profile["bio"]); ids.append("bio"); metadatas.append({"source": "bio"})

    if os.path.exists("data/cv.pdf"):
        reader = PdfReader("data/cv.pdf")
        full_text = "\n".join(p.extract_text() or "" for p in reader.pages)
        for i, chunk in enumerate(chunk_text(full_text)):
            docs.append(chunk); ids.append(f"cv-{i}"); metadatas.append({"source": "cv"})

    embeddings = embed_texts(docs, task_type="RETRIEVAL_DOCUMENT")
    collection.add(documents=docs, ids=ids, metadatas=metadatas, embeddings=embeddings)
    print(f"Index built: {len(docs)} chunks")

def answer_question(question: str, n_results: int = 4) -> str:
    query_embedding = embed_texts([question], task_type="RETRIEVAL_QUERY")[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)
    context = "\n\n---\n\n".join(results["documents"][0])

    response = client_ai.models.generate_content(
        model="gemini-1.5-flash",
        contents=f"Context:\n{context}\n\nQuestion: {question}",
        config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
    )
    return response.text