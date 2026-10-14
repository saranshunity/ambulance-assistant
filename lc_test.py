import os
from dotenv import load_dotenv

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY", "")

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from slots import BookingSlots
import tools as T

llm = ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL"), temperature=0)

tool_list = [tool(f) for f in (
    T.find_hospitals, T.list_ambulance_types, T.estimate_fare,
    T.prepare_booking, T.create_booking, T.emergency_dispatch,
)]
print("Tools:", [t.name for t in tool_list])

# 1. Tool calling: the model only *asks* for a tool
ai = llm.bind_tools(tool_list).invoke("Which hospitals are near Green Park?")
print("Tool calls:", ai.tool_calls)

# 2. Structured output: the same idea as your extract.py
structured = llm.with_structured_output(BookingSlots)
print(structured.invoke("Mera naam Rahul hai, Panipat bus stand se ICU ambulance chahiye."))