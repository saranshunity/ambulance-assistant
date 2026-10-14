import argparse
import json
import os
import time
import uuid

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import tools as T
from evals.cases import CASES
from graph_agent import graph, message_text, _langfuse_client

parser = argparse.ArgumentParser()
parser.add_argument("label")
parser.add_argument("--only", default="", help="comma-separated case ids")
parser.add_argument("--pause", type=float, default=5.0, help="seconds between turns")
args = parser.parse_args()

os.makedirs("evals/runs", exist_ok=True)
RUN_FILE = f"evals/runs/{args.label}.jsonl"


def norm(v):
    return str(v).strip().lower()


def collect_events(messages):
    results = {}
    for m in messages:
        if isinstance(m, ToolMessage):
            try:
                results[m.tool_call_id] = json.loads(m.content)
            except Exception:
                results[m.tool_call_id] = {"raw": str(m.content)}
    events = []
    for m in messages:
        if isinstance(m, AIMessage):
            for c in (m.tool_calls or []):
                events.append({"name": c["name"], "args": c["args"],
                               "result": results.get(c["id"], {})})
    return events


def run_checks(expect, final, events, reply):
    fails = []
    names = [e["name"] for e in events]
    reply_l = reply.lower()
    slots = final.get("slots") or {}

    for t in expect.get("tools_called", []):
        if t not in names:
            fails.append(f"tool {t} was not called")
    for t in expect.get("tools_not_called", []):
        if t in names:
            fails.append(f"tool {t} should not have been called")

    if "booked" in expect:
        booked = any(e["name"] == "create_booking" and "booking_id" in e["result"] for e in events)
        if booked != expect["booked"]:
            fails.append(f"booked={booked}, expected {expect['booked']}")
    if "emergency_dispatched" in expect:
        sent = any(e["name"] == "emergency_dispatch" and "booking_id" in e["result"] for e in events)
        if sent != expect["emergency_dispatched"]:
            fails.append(f"emergency_dispatched={sent}, expected {expect['emergency_dispatched']}")
    if "handoff" in expect:
        got = bool(final.get("handoff"))
        if got != expect["handoff"]:
            fails.append(f"handoff={got}, expected {expect['handoff']}")

    for tool_name, want in expect.get("args_contain", {}).items():
        ok = any(e["name"] == tool_name and
                 all(norm(e["args"].get(k, "")) == norm(v) for k, v in want.items())
                 for e in events)
        if not ok:
            fails.append(f"no {tool_name} call with arguments {want}")
    for k, v in expect.get("slots_equal", {}).items():
        if norm(slots.get(k, "")) != norm(v):
            fails.append(f"slot {k}={slots.get(k)!r}, expected {v!r}")
    for k in expect.get("slots_absent", []):
        if slots.get(k):
            fails.append(f"slot {k} should be empty but is {slots.get(k)!r}")

    any_of = expect.get("reply_contains_any")
    if any_of and not any(s.lower() in reply_l for s in any_of):
        fails.append(f"reply contains none of {any_of}")
    for s in expect.get("reply_not_contains", []):
        if s.lower() in reply_l:
            fails.append(f"reply contains forbidden text {s!r}")
    return fails


def is_transient(e):
    s = str(e)
    return any(x in s for x in ("429", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE"))


def play(case, config):
    final = None
    for text in case["turns"]:
        final = graph.invoke({"messages": [HumanMessage(content=text)]}, config=config)
        time.sleep(args.pause)
    return final


def run_case(case, label):
    langfuse = _langfuse_client()
    final, trace_id = None, None
    for attempt in range(3):
        T.reset_fleet()
        thread_id = f"eval-{label}-{case['id']}-{uuid.uuid4().hex[:6]}"
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 16}
        try:
            if langfuse:
                from langfuse import propagate_attributes
                from langfuse.langchain import CallbackHandler
                with langfuse.start_as_current_observation(as_type="span", name=f"eval:{case['id']}"):
                    with propagate_attributes(session_id=thread_id,
                                              tags=["eval", label, case["group"]]):
                        trace_id = langfuse.get_current_trace_id()
                        config["callbacks"] = [CallbackHandler()]
                        final = play(case, config)
            else:
                final = play(case, config)
            break
        except Exception as e:
            if attempt < 2 and is_transient(e):
                print(f"  [rate limit or overload, waiting 65s] {str(e)[:100]}")
                time.sleep(65)
                continue
            return {"id": case["id"], "error": str(e)[:300]}

    messages = final["messages"]
    events = collect_events(messages)
    reply = message_text(messages[-1])
    fails = run_checks(case["expect"], final, events, reply)

    if langfuse and trace_id:
        langfuse.create_score(trace_id=trace_id, name="eval-pass",
                              value=0.0 if fails else 1.0, data_type="NUMERIC",
                              comment="; ".join(fails)[:500] or None)
    return {"id": case["id"], "group": case["group"], "passed": not fails,
            "failures": fails, "reply": reply[:300],
            "tools": [e["name"] for e in events]}


# ---- run, resuming from any earlier results for this label ----
done = {}
if os.path.exists(RUN_FILE):
    with open(RUN_FILE, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            done[r["id"]] = r

only = {x.strip() for x in args.only.split(",") if x.strip()}
todo = [c for c in CASES if c["id"] not in done and (not only or c["id"] in only)]
print(f"{len(done)} cases already done, {len(todo)} to run")

for case in todo:
    print(f"\n[{case['group']}] {case['id']}")
    result = run_case(case, args.label)
    if "error" in result:
        print("  ERROR (will be retried on the next run):", result["error"])
        continue
    done[case["id"]] = result
    with open(RUN_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
    print("  PASS" if result["passed"] else "  FAIL: " + "; ".join(result["failures"]))

# ---- summary ----
results = [done[c["id"]] for c in CASES if c["id"] in done]
if results:
    passed = sum(r["passed"] for r in results)
    groups = {}
    for r in results:
        g = groups.setdefault(r["group"], [0, 0])
        g[0] += r["passed"]
        g[1] += 1
    summary = (f"[{args.label}] {passed}/{len(results)} passed | " +
               " | ".join(f"{g} {a}/{b}" for g, (a, b) in groups.items()))
    print("\n" + summary)
    for r in results:
        if not r["passed"]:
            print(f"\nFAILED {r['id']}: {'; '.join(r['failures'])}")
            print(f"  tools: {r['tools']}")
            print(f"  reply: {r['reply'][:200]}")
    with open("evals/log.txt", "a", encoding="utf-8") as f:
        f.write(summary + "\n")

lf = _langfuse_client()
if lf:
    lf.flush()