import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
import tools
from retry import with_retry

load_dotenv()
client = genai.Client()
MODEL = os.getenv("GEMINI_MODEL")

SYSTEM = """You are a calm, brief ambulance-booking assistant for a demo service.
- Reply in the user's language (Hindi, English or Hinglish). Keep replies short and ask one question at a time.
- Never give medical advice or diagnose.
- Needed to book: patient name, phone, pickup area, hospital, ambulance type (basic, oxygen or icu).
- Get distances, hospital lists, fares and ETAs ONLY from tools. Never invent numbers.
- If an area is unknown, show the known areas the tool returns.
- When all details are known, call prepare_booking, show the summary, and ask the user to confirm.
- Call create_booking only after the user confirms in a LATER message.
- After booking, tell the user the booking id, ambulance, ETA and fare.
- If the user describes an emergency, tell them to call 112 first, then ask only for the pickup area.
- EMERGENCY (collapse, accident, unconscious, breathing trouble, heavy bleeding): do not ask for name, phone or confirmation. If the pickup area is unknown, ask only for it. As soon as you have it, call emergency_dispatch, then give the ambulance id, ETA and nearest hospital in one or two short sentences. Never judge how serious the situation is.
- Reply in the same script the user wrote in (Devanagari if they wrote Devanagari, Latin letters if they wrote Latin)."""

EMERGENCY_BANNER = "IMPORTANT: Call 112 now if you haven't already. This is a demo service."
CONFIG = types.GenerateContentConfig(
    system_instruction=SYSTEM,
    tools=[tools.find_hospitals, tools.list_ambulance_types, tools.estimate_fare,
           tools.prepare_booking, tools.create_booking, tools.emergency_dispatch],
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    temperature=0.2,
)

SAFE_TOOLS = {
    "find_hospitals": tools.find_hospitals,
    "list_ambulance_types": tools.list_ambulance_types,
    "estimate_fare": tools.estimate_fare,
}

sessions: dict[str, dict] = {}


def run_tool(name: str, args: dict, state: dict) -> dict:
    try:
        if name in SAFE_TOOLS:
            return SAFE_TOOLS[name](**args)

        if name == "prepare_booking":
            result = tools.prepare_booking(**args)
            if "error" not in result:
                state["prepared"] = {"args": args, "turn": state["turn"]}
            return result

        if name == "create_booking":
            prepared = state["prepared"]
            if prepared is None or prepared["turn"] == state["turn"]:
                return {"error": "needs_user_confirmation",
                        "message": "Show the summary and wait for the user to confirm in their next message."}
            if prepared["args"] != args:
                return {"error": "details_changed",
                        "message": "Details differ from the confirmed summary. Prepare again."}
            result = tools.create_booking(**args)
            if "error" not in result:
                state["prepared"] = None
            return result
        if name == "emergency_dispatch":
            if state.get("emergency_done"):
                return {"error": "already_dispatched"}
            result = tools.emergency_dispatch(**args)
            if "error" not in result:
                state["emergency_done"] = True
                state["emergency_flag"] = True
            return result
        return {"error": "unknown_tool"}
    except TypeError as e:
        return {"error": "bad_arguments", "detail": str(e)}


def chat_turn(session_id: str, message: str) -> str:
    state = sessions.setdefault(session_id, {"history": [], "turn": 0, "prepared": None})
    state["turn"] += 1
    state["emergency_flag"] = False
    state["history"].append(types.Content(role="user", parts=[types.Part(text=message)]))
    

    for _ in range(6):                      # safety cap on tool rounds
        response = with_retry(lambda: client.models.generate_content(
            model=MODEL, contents=state["history"], config=CONFIG
        ))
        model_content = response.candidates[0].content
        state["history"].append(model_content)

        calls = response.function_calls
        if not calls:
            reply = response.text or "Sorry, could you repeat that?"
            if state.get("emergency_flag"):
                reply = EMERGENCY_BANNER + "\n\n" + reply
            return reply
        parts = []
        for call in calls:
            args = dict(call.args or {})
            result = run_tool(call.name, args, state)
            print(f"[tool] {call.name}({args}) -> {result}")
            parts.append(types.Part.from_function_response(
                name=call.name, response={"result": result}))
        state["history"].append(types.Content(role="user", parts=parts))

    return "Sorry, I couldn't complete that. Please try again."