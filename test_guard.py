from agent import run_tool

args = dict(patient_name="Rahul", phone="9999999999", pickup_area="Green Park",
            hospital_name="Demo City Hospital", ambulance_type="icu")

state = {"history": [], "turn": 1, "prepared": None}
print("1. book with no prepare:   ", run_tool("create_booking", args, state))
run_tool("prepare_booking", args, state)
print("2. book in the same turn:  ", run_tool("create_booking", args, state))
state["turn"] = 2
print("3. next turn, changed type:", run_tool("create_booking", {**args, "ambulance_type": "basic"}, state))
print("4. next turn, same details:", run_tool("create_booking", args, state))
print("5. book again:             ", run_tool("create_booking", args, state))