from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.rag import answer_question, build_index

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-portfolio-domain.com", "http://localhost:3000"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup():
    build_index()  # rebuilds the in-memory index every time the server boots

class ChatRequest(BaseModel):
    question: str

@app.post("/chat")
def chat(req: ChatRequest):
    return {"answer": answer_question(req.question)}