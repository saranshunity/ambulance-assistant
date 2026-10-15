import traceback
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from google.genai import errors
from pydantic import BaseModel
from graph_agent import chat_turn_graph, record_feedback
from referral import extract_referral, MAX_BYTES

app = FastAPI(title="Ambulance Assistant")


class AgentRequest(BaseModel):
    session_id: str
    message: str


class FeedbackRequest(BaseModel):
    trace_id: str
    value: int


@app.get("/")
def home():
    return FileResponse("static/index.html")


@app.post("/agent")
def agent_endpoint(req: AgentRequest):
    try:
        reply, slots, trace_id = chat_turn_graph(req.session_id, req.message)
        return {"reply": reply, "slots": slots, "trace_id": trace_id}
    except Exception:
        traceback.print_exc()
        raise HTTPException(status_code=503, detail="The assistant is busy, please try again.")


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    if req.value not in (0, 1):
        raise HTTPException(status_code=400, detail="value must be 0 or 1")
    return {"saved": record_feedback(req.trace_id, req.value)}


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