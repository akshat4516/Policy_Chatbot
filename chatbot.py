import os
import uuid
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import StrOutputParser

# Load PDFs
pdf_folder = "data"
pdf_files = [os.path.join(pdf_folder, f) for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
all_pages = []
for file in pdf_files:
    loader = PDFPlumberLoader(file)
    pages = loader.load()
    all_pages.extend(loader.load())

# Chunking
splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=500)
chunks = splitter.split_documents(all_pages)

# Embedding and Vectorstore
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embedding)
retriever = vectorstore.as_retriever(k=3)

# LLM setup
prompt = PromptTemplate(
    input_variables=["chat_history", "question", "documents"],
    template="""
You are a helpful and professional assistant. Use only the information from the documents below and the ongoing conversation.

✅ Do not mention document names, sections, or numbers.
✅ Do not say phrases like “according to the document” or “based on section...”
✅ Provide clear, human-like answers as if you are responding directly to the user in a real-time chat.
✅ Focus on clarity and completeness.

Chat History:
{chat_history}

User Question:
{question}

Relevant Info:
{documents}

Final Answer:
"""
)

llm = ChatOllama(model="llama3", temperature=0)
chat_chain = prompt | llm | StrOutputParser()

# Memory and Session Management
class ChatRAGApplication:
    def __init__(self, retriever, chat_chain):
        self.session_id = str(uuid.uuid4())
        self.retriever = retriever
        self.chat_chain = chat_chain
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=False)

    async def chat(self, question: str) -> str:
        self.memory.chat_memory.add_user_message(question)
        documents = self.retriever.invoke(question)
        doc_texts = "\n".join([doc.page_content for doc in documents])
        print("📄 Retrieved Documents for Debugging:\n", doc_texts)
        chat_history = self.memory.load_memory_variables({}).get("chat_history", "")
        prompt_vars = {
            "chat_history": chat_history,
            "question": question,
            "documents": doc_texts
        }
        answer = self.chat_chain.invoke(prompt_vars)
        self.memory.chat_memory.add_ai_message(answer)
        return answer

    def get_history(self) -> str:
        return self.memory.load_memory_variables({}).get("chat_history", "")

    def get_details(self) -> dict:
        history = self.get_history()
        return {
            "session_id": self.session_id,
            "conversation_length": len([line for line in history.split("\n") if line.strip()])
        }

sessions = {}
def get_session(session_id: str) -> ChatRAGApplication:
    if session_id not in sessions:
        sessions[session_id] = ChatRAGApplication(retriever, chat_chain)
    return sessions[session_id]

# FastAPI setup
app = FastAPI()

class ChatRequest(BaseModel):
    session_id: str = None
    message: str

class ChatResponse(BaseModel):
    session_id: str
    answer: str
    history: str
    session_details: dict

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if not request.session_id:
        session = ChatRAGApplication(retriever, chat_chain)
        sessions[session.session_id] = session
    else:
        session = get_session(request.session_id)
    answer = await session.chat(request.message)
    return ChatResponse(
        session_id=session.session_id,
        answer=answer,
        history=session.get_history(),
        session_details=session.get_details()
    )

@app.delete("/session/{session_id}", response_model=dict)
def end_session(session_id: str):
    if session_id in sessions:
        del sessions[session_id]
        return {"detail": f"Session {session_id} ended."}
    raise HTTPException(status_code=404, detail="Session not found.")
