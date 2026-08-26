import os
import json
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
import google.generativeai as genai

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
client = chromadb.EphemeralClient()  # in-memory, no disk needed
collection = client.get_or_create_collection("portfolio", embedding_function=embed_fn)

SYSTEM_PROMPT = """You are a helpful assistant answering questions about Rabin Patel \
for recruiters and visitors to his portfolio site. Only answer using the provided \
context. If the context doesn't contain the answer, say you don't have that \
information and suggest they reach out directly via email. Keep answers concise \
and professional. Speak about Rabin in the third person."""

def chunk_text(text: str, size: int = 500, overlap: int = 50) -> list[str]:
    words = text.split()
    return [" ".join(words[i : i + size]) for i in range(0, len(words), size - overlap)]

def build_index():
    """Runs once at server startup — rebuilds the in-memory vector index."""
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

    collection.add(documents=docs, ids=ids, metadatas=metadatas)
    print(f"Index built: {len(docs)} chunks")

def answer_question(question: str, n_results: int = 4) -> str:
    results = collection.query(query_texts=[question], n_results=n_results)
    context = "\n\n---\n\n".join(results["documents"][0])

    model = genai.GenerativeModel(
        "gemini-1.5-flash",
        system_instruction=SYSTEM_PROMPT,
    )
    response = model.generate_content(f"Context:\n{context}\n\nQuestion: {question}")
    return response.text