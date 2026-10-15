from graph_agent import execute

args = dict(patient_name="Rahul", phone="9999999999", pickup_area="Green Park",
            hospital_name="Demo City Hospital", ambulance_type="icu")


def call(name, a, state):
    result, updates = execute(name, a, state)
    state.update(updates)
    return result


state = {"turn": 1, "prepared": None}
print("1. book with no prepare:   ", call("create_booking", args, state))
call("prepare_booking", args, state)
print("2. book in the same turn:  ", call("create_booking", args, state))
state["turn"] = 2
print("3. next turn, changed type:", call("create_booking", {**args, "ambulance_type": "basic"}, state))
print("4. next turn, same details:", call("create_booking", args, state))
print("5. book again:             ", call("create_booking", args, state))