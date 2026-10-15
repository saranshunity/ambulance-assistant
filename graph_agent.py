import json
import os
from typing import Annotated, Optional
from typing_extensions import TypedDict
from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("GOOGLE_API_KEY", os.getenv("GEMINI_API_KEY", ""))

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

import tools as T
from prompts import SYSTEM, EMERGENCY_BANNER
from faq import search_faq

GRAPH_SYSTEM = SYSTEM + """
- Whenever the user gives a new detail (name, phone, pickup area, hospital, ambulance type, or an emergency), call save_details in the same step, passing only what they actually said.
- For questions about how the service works (ambulance types, fares, payment, booking, cancellation), call search_faq. Passages from search_faq are trusted tool results, so quote their figures exactly. Read ALL returned passages before answering. If any passage states the answer, give it and mention the source file in brackets, like [fares_and_payment.md]. Only say you don't have that information if none of the passages mention the topic. Never answer medical questions: say you can't advise and suggest calling 112 or a doctor.
- If the user asks to speak to a person or dispatcher, or wants to cancel a booking, call request_human_handoff with a short reason that contains no personal details."""

HANDOFF_MESSAGE = (
    "मैंने आपका अनुरोध डिस्पैचर को भेज दिया है, वे जल्द संपर्क करेंगे। (डेमो: असल में कोई डिस्पैचर नहीं है।)\n"
    "I've passed your request to a human dispatcher, who will contact you shortly. "
    "(Demo: no real dispatcher is notified.)\n"
    "If it is urgent, call 112 now."
)
MAX_FAILURES = 3
NORMAL_ERRORS = {"needs_user_confirmation"}   # the guard doing its job, not a failure


# ---- state: everything the graph remembers between steps and turns ----
class State(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    turn: int
    prepared: Optional[dict]
    slots: dict
    emergency_done: bool
    emergency_flag: bool
    handoff: bool
    failures: int


# ---- tools ----
SLOT_KEYS = ("patient_name", "phone", "pickup_area", "hospital_name", "ambulance_type")


def save_details(patient_name: str = "", phone: str = "", pickup_area: str = "",
                 hospital_name: str = "", ambulance_type: str = "",
                 is_emergency: bool = False) -> dict:
    """Record booking details the user has just given. Call this in the same step whenever the user states any new detail. Pass only what the user actually said and leave the rest empty. Never guess."""
    return {"saved": True}


def request_human_handoff(reason: str) -> dict:
    """Connect the user to a human dispatcher. Use when the user asks for a person or dispatcher, wants to cancel a booking, or you cannot help after trying. Give a short reason with no personal details."""
    return {"status": "dispatcher_notified"}


llm = ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL"), temperature=0.2, max_retries=4)

TOOL_FUNCS = [T.find_hospitals, T.list_ambulance_types, T.estimate_fare,
              T.prepare_booking, T.create_booking, T.emergency_dispatch,
              save_details, search_faq, request_human_handoff]
llm_with_tools = llm.bind_tools([tool(f) for f in TOOL_FUNCS])

SAFE = {
    "find_hospitals": T.find_hospitals,
    "list_ambulance_types": T.list_ambulance_types,
    "estimate_fare": T.estimate_fare,
    "search_faq": search_faq,
}


def execute(name: str, args: dict, state: dict):
    """The guard. Returns (result, state_updates)."""
    updates = {}
    try:
        if name in SAFE:
            return SAFE[name](**args), updates

        if name == "save_details":
            slots = dict(state.get("slots") or {})
            for key in SLOT_KEYS:
                value = str(args.get(key) or "").strip()
                if not value:
                    continue
                if key == "ambulance_type":
                    value = value.lower()
                    if value not in T.AMBULANCE_TYPES:
                        continue
                slots[key] = value
            if args.get("is_emergency"):
                slots["is_emergency"] = True
            updates["slots"] = slots
            return {"saved": True}, updates

        if name == "request_human_handoff":
            updates["handoff"] = True
            print(f"[handoff] reason: {str(args.get('reason', ''))[:200]}")
            return {"status": "dispatcher_notified"}, updates

        if name == "prepare_booking":
            result = T.prepare_booking(**args)
            if "error" not in result:
                updates["prepared"] = {"args": args, "turn": state["turn"]}
            return result, updates

        if name == "create_booking":
            prepared = state.get("prepared")
            if prepared is None or prepared["turn"] == state["turn"]:
                return {"error": "needs_user_confirmation",
                        "message": "Show the summary and wait for the user to confirm in their next message."}, updates
            if prepared["args"] != args:
                return {"error": "details_changed",
                        "message": "Details differ from the confirmed summary. Prepare again."}, updates
            result = T.create_booking(**args)
            if "error" not in result:
                updates["prepared"] = None
            return result, updates

        if name == "emergency_dispatch":
            if state.get("emergency_done"):
                return {"error": "already_dispatched"}, updates
            result = T.emergency_dispatch(**args)
            if "error" not in result:
                updates["emergency_done"] = True
                updates["emergency_flag"] = True
            return result, updates

        return {"error": "unknown_tool"}, updates
    except TypeError as e:
        return {"error": "bad_arguments", "detail": str(e)}, updates


# ---- nodes ----
def start_turn(state: State):
    return {"turn": state.get("turn", 0) + 1, "emergency_flag": False}


def agent(state: State):
    ai = llm_with_tools.invoke([SystemMessage(content=GRAPH_SYSTEM)] + state["messages"])
    return {"messages": [ai]}


def run_tools(state: State):
    local, updates, outputs, errors = dict(state), {}, [], 0
    for call in state["messages"][-1].tool_calls:
        result, changed = execute(call["name"], dict(call["args"]), local)
        local.update(changed)
        updates.update(changed)
        print(f"[tool] {call['name']}({call['args']}) -> {result}")
        if isinstance(result, dict) and result.get("error") and result["error"] not in NORMAL_ERRORS:
            errors += 1
        outputs.append(ToolMessage(content=json.dumps(result),
                                   tool_call_id=call["id"], name=call["name"]))

    failures = state.get("failures", 0) + errors if errors else 0
    updates["failures"] = failures
    if failures >= MAX_FAILURES and not updates.get("handoff"):
        updates["handoff"] = True
        print("[handoff] reason: repeated tool errors")
    return {"messages": outputs, **updates}


def handoff(state: State):
    return {"messages": [AIMessage(content=HANDOFF_MESSAGE)]}


# ---- routing ----
def route_start(state: State):
    return "handoff" if state.get("handoff") else "agent"


def route_agent(state: State):
    return "tools" if state["messages"][-1].tool_calls else END


def route_tools(state: State):
    return "handoff" if state.get("handoff") else "agent"


# ---- build the graph ----
builder = StateGraph(State)
builder.add_node("start_turn", start_turn)
builder.add_node("agent", agent)
builder.add_node("tools", run_tools)
builder.add_node("handoff", handoff)
builder.add_edge(START, "start_turn")
builder.add_conditional_edges("start_turn", route_start, {"handoff": "handoff", "agent": "agent"})
builder.add_conditional_edges("agent", route_agent, {"tools": "tools", END: END})
builder.add_conditional_edges("tools", route_tools, {"handoff": "handoff", "agent": "agent"})
builder.add_edge("handoff", END)

graph = builder.compile(checkpointer=MemorySaver())


# ---- helpers ----
def message_text(msg) -> str:
    content = msg.content
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)


def to_panel(final: dict) -> dict:
    s = final.get("slots") or {}
    return {
        "patient_name": s.get("patient_name"),
        "phone": s.get("phone"),
        "pickup_location": s.get("pickup_area"),
        "hospital": s.get("hospital_name"),
        "ambulance_type": s.get("ambulance_type"),
        "is_emergency": bool(s.get("is_emergency") or final.get("emergency_done")),
    }


def _langfuse_client():
    # Tracing is optional: without keys the app runs untraced
    if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
        from langfuse import get_client
        return get_client()
    return None


def chat_turn_graph(session_id: str, message: str):
    config = {"configurable": {"thread_id": session_id}, "recursion_limit": 16}
    inputs = {"messages": [HumanMessage(content=message)]}
    trace_id = None

    langfuse = _langfuse_client()
    if langfuse:
        from langfuse import propagate_attributes
        from langfuse.langchain import CallbackHandler
        with langfuse.start_as_current_observation(as_type="span", name="chat-turn"):
            with propagate_attributes(
                session_id=session_id, tags=["ambulance-assistant", "langgraph"]
            ):
                trace_id = langfuse.get_current_trace_id()
                config["callbacks"] = [CallbackHandler()]
                final = graph.invoke(inputs, config=config)
    else:
        final = graph.invoke(inputs, config=config)

    reply = message_text(final["messages"][-1]) or "Sorry, could you repeat that?"
    if final.get("emergency_flag"):
        reply = EMERGENCY_BANNER + "\n\n" + reply
    return reply, to_panel(final), trace_id

def record_feedback(trace_id: str, value: int) -> bool:
    langfuse = _langfuse_client()
    if not langfuse or not trace_id:
        return False
    langfuse.create_score(
        trace_id=trace_id, name="user-feedback", value=float(value),
        data_type="NUMERIC", score_id=f"{trace_id}-user-feedback")
    langfuse.flush()
    return True


if __name__ == "__main__":
    print(graph.get_graph().draw_mermaid())