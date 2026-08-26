import os
import json
import chromadb
from pypdf import PdfReader
from google import genai
from google.genai import types

# 1. Base Environment and Client Configuration
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY is not set — check your .env file or host environment variables")

# Initialize default clean SDK setup (routes natively via standard v1beta endpoints)
client_ai = genai.Client(api_key=api_key)

# 2. Vector Database Initialization
chroma_client = chromadb.EphemeralClient()
collection = chroma_client.get_or_create_collection("portfolio")

# 3. AI Directives
SYSTEM_PROMPT = """You are a helpful assistant answering questions about Rabin Patel \
for recruiters and visitors to his portfolio site. Only answer using the provided \
context. If the context doesn't contain the answer, say you don't have that \
information and suggest they reach out directly via email.

Keep answers concise but always complete your thought — don't cut off mid-sentence. \
Aim for 4 to 8 sentences for most questions. When listing multiple items (e.g. skills, \
projects), list them clearly rather than cramming everything into one sentence. \
Speak about Rabin in the third person."""

# 4. Core RAG Utility Functions
def embed_texts(texts: list[str], task_type: str) -> list[list[float]]:
    """Generate one embedding for each text using Gemini Embedding 2."""
    all_embeddings = []

    for text in texts:
        result = client_ai.models.embed_content(
            model="gemini-embedding-2",
            contents=text,
            config=types.EmbedContentConfig(
                task_type=task_type
            ),
        )

        if not result.embeddings:
            raise RuntimeError("Gemini returned no embedding.")

        embedding = result.embeddings[0]

        if not embedding.values:
            raise RuntimeError("Gemini returned an empty embedding.")

        all_embeddings.append(embedding.values)

    return all_embeddings

def chunk_text(text: str, size: int = 500, overlap: int = 50) -> list[str]:
    """Splits bulky unstructured content into overlapping, searchable tokens."""
    words = text.split()
    return [" ".join(words[i : i + size]) for i in range(0, len(words), size - overlap)]

def build_index():
    """Compiles profile schemas and localized CV texts into a synchronized vector space."""
    docs, ids, metadatas = [], [], []

    # Parse primary JSON data
    with open("data/profile.json") as f:
        profile = json.load(f)

    for i, job in enumerate(profile["experience"]):
        text = f"{job['role']} at {job['company']} ({job['dates']}). {job['summary']} " + " ".join(job["bullets"])
        docs.append(text)
        ids.append(f"exp-{i}")
        metadatas.append({"source": "experience"})

    for i, proj in enumerate(profile["projects"]):
        text = f"Project: {proj['title']}. {proj['problem']} {proj['description']} Result: {proj['result']}"
        docs.append(text)
        ids.append(f"proj-{i}")
        metadatas.append({"source": "project"})

    docs.append(profile["bio"])
    ids.append("bio")
    metadatas.append({"source": "bio"})

    # Optional step: Unpack system CV PDFs if attached to workspace
    if os.path.exists("data/cv.pdf"):
        reader = PdfReader("data/cv.pdf")
        full_text = "\n".join(p.extract_text() or "" for p in reader.pages)
        for i, chunk in enumerate(chunk_text(full_text)):
            docs.append(chunk)
            ids.append(f"cv-{i}")
            metadatas.append({"source": "cv"})

    # CRITICAL FIX: Aligned explicitly to baseline function level (4 spaces)
    embeddings = embed_texts(docs, task_type="RETRIEVAL_DOCUMENT")
    collection.add(documents=docs, ids=ids, metadatas=metadatas, embeddings=embeddings)
    print(f"Index built: {len(docs)} chunks")

def answer_question(question: str, n_results: int = 4) -> str:
    """Queries vector storage and generates a contextualized response."""

    query_embedding = embed_texts(
        [question],
        task_type="RETRIEVAL_QUERY"
    )[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    context = "\n\n---\n\n".join(results["documents"][0])

    response = client_ai.models.generate_content(
        model="gemini-3.6-flash",
        contents=f"Context:\n{context}\n\nQuestion: {question}",
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=450,
        ),
    )

    return response.text