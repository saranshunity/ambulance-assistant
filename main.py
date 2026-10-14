import traceback
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from google.genai import errors
from agent import chat_turn
from llm import stream_answer
from slots import BookingSlots
from extract import extract_slots
from referral import extract_referral, MAX_BYTES
from graph_agent import chat_turn_graph, record_feedback

app = FastAPI(title="Ambulance Assistant")

SYSTEM = """You are a calm, brief ambulance-booking assistant.
- Reply in the same language the user writes in (Hindi, English or Hinglish).
- Never give medical advice or diagnose.
- If the user describes an emergency, tell them to call 112 right away."""

# In-memory store: session_id -> booking details collected so far
sessions: dict[str, BookingSlots] = {}
class FeedbackRequest(BaseModel):
    trace_id: str
    value: int

class ChatRequest(BaseModel):
    message: str


class ExtractRequest(BaseModel):
    session_id: str
    message: str


@app.get("/")
def home():
    return FileResponse("static/index.html")


@app.post("/chat")
def chat(req: ChatRequest):
    prompt = f"{SYSTEM}\n\nUser: {req.message}\nAssistant:"
    return StreamingResponse(stream_answer(prompt), media_type="text/plain")


@app.post("/extract")
def extract(req: ExtractRequest):
    current = sessions.get(req.session_id, BookingSlots())
    updated = extract_slots(req.message, current)
    sessions[req.session_id] = updated
    return updated

class AgentRequest(BaseModel):
    session_id: str
    message: str


@app.post("/agent")
def agent_endpoint(req: AgentRequest):
    try:
        return {"reply": chat_turn(req.session_id, req.message)}
    except errors.APIError:
        raise HTTPException(status_code=503, detail="The assistant is busy, please try again.")

@app.post("/referral")
def referral(file: UploadFile = File(...)):
    data = file.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 5 MB).")
    if not data.startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")
    try:
        return extract_referral(data)
    except errors.APIError:
        raise HTTPException(status_code=503, detail="Could not read the document, please try again.")

@app.post("/agent2")
def agent2_endpoint(req: AgentRequest):
    try:
        reply, slots, trace_id = chat_turn_graph(req.session_id, req.message)
        return {"reply": reply, "slots": slots, "trace_id": trace_id}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=503, detail="The assistant is busy, please try again.")


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    if req.value not in (0, 1):
        raise HTTPException(status_code=400, detail="value must be 0 or 1")
    return {"saved": record_feedback(req.trace_id, req.value)}