import math
import uuid

# Invented demo data. Positions are (x, y) in km on a made-up grid.
AREAS = {
    "green park": (2, 3),
    "station road": (5, 1),
    "model colony": (8, 6),
    "sector 12": (11, 2),
    "old market": (4, 7),
}

HOSPITALS = [
    {"name": "Demo City Hospital", "pos": (3, 4), "has_icu": True},
    {"name": "Riverside Care Centre", "pos": (9, 5), "has_icu": False},
    {"name": "Sunrise Medical Institute", "pos": (6, 2), "has_icu": True},
]

AMBULANCE_TYPES = {
    "basic": {"base_fare": 500, "per_km": 15},
    "oxygen": {"base_fare": 800, "per_km": 20},
    "icu": {"base_fare": 2000, "per_km": 40},
}

FLEET = [
    {"id": "AMB-1", "type": "basic", "pos": (1, 1), "free": True},
    {"id": "AMB-2", "type": "basic", "pos": (7, 4), "free": True},
    {"id": "AMB-3", "type": "oxygen", "pos": (3, 6), "free": True},
    {"id": "AMB-4", "type": "icu", "pos": (5, 3), "free": True},
    {"id": "AMB-5", "type": "icu", "pos": (10, 3), "free": True},
]


def _area(name: str):
    return AREAS.get(name.strip().lower())


def _hospital(name: str):
    key = name.strip().lower()
    return next((h for h in HOSPITALS if h["name"].lower() == key), None)


def find_hospitals(pickup_area: str) -> dict:
    """Find hospitals near the pickup area, nearest first, with distance in km and whether they have an ICU."""
    pos = _area(pickup_area)
    if pos is None:
        return {"error": "unknown_area", "known_areas": list(AREAS)}
    found = [
        {"name": h["name"], "distance_km": round(math.dist(pos, h["pos"]), 1), "has_icu": h["has_icu"]}
        for h in HOSPITALS
    ]
    found.sort(key=lambda h: h["distance_km"])
    return {"hospitals": found}


def list_ambulance_types() -> dict:
    """List the ambulance types (basic, oxygen, icu) with base fare and per-km rate in rupees."""
    return {"types": AMBULANCE_TYPES}


def estimate_fare(pickup_area: str, hospital_name: str, ambulance_type: str) -> dict:
    """Estimate the fare in rupees for a trip from the pickup area to a hospital with the given ambulance type."""
    pos, hospital = _area(pickup_area), _hospital(hospital_name)
    if pos is None or hospital is None or ambulance_type not in AMBULANCE_TYPES:
        return {"error": "invalid_details"}
    km = round(math.dist(pos, hospital["pos"]), 1)
    rate = AMBULANCE_TYPES[ambulance_type]
    return {"distance_km": km, "fare_inr": round(rate["base_fare"] + rate["per_km"] * km)}


def prepare_booking(patient_name: str, phone: str, pickup_area: str,
                    hospital_name: str, ambulance_type: str) -> dict:
    """Check the booking details and return a summary with the fare. Call this once ALL details are known, then ask the user to confirm. This does NOT book anything."""
    pos, hospital = _area(pickup_area), _hospital(hospital_name)
    if pos is None or hospital is None or ambulance_type not in AMBULANCE_TYPES:
        return {"error": "invalid_details", "known_areas": list(AREAS)}
    est = estimate_fare(pickup_area, hospital_name, ambulance_type)
    return {
        "status": "awaiting_user_confirmation",
        "summary": {
            "patient_name": patient_name, "phone": phone,
            "pickup_area": pickup_area, "hospital": hospital["name"],
            "ambulance_type": ambulance_type,
            "distance_km": est["distance_km"], "fare_inr": est["fare_inr"],
        },
    }


def create_booking(patient_name: str, phone: str, pickup_area: str,
                   hospital_name: str, ambulance_type: str) -> dict:
    """Book the ambulance. Only works after prepare_booking was called in an EARLIER message and the user confirmed. Use exactly the same details."""
    pos, hospital = _area(pickup_area), _hospital(hospital_name)
    if pos is None or hospital is None or ambulance_type not in AMBULANCE_TYPES:
        return {"error": "invalid_details"}
    free = [a for a in FLEET if a["type"] == ambulance_type and a["free"]]
    if not free:
        return {"error": "no_ambulance_available", "type": ambulance_type}
    amb = min(free, key=lambda a: math.dist(a["pos"], pos))
    amb["free"] = False
    eta = max(2, round(math.dist(amb["pos"], pos) / 30 * 60))   # assume 30 km/h
    fare = estimate_fare(pickup_area, hospital_name, ambulance_type)["fare_inr"]
    return {
        "booking_id": uuid.uuid4().hex[:8].upper(),
        "ambulance_id": amb["id"],
        "eta_minutes": eta,
        "fare_inr": fare,
        "patient_name": patient_name,
        "pickup_area": pickup_area,
        "hospital": hospital["name"],
    }

def emergency_dispatch(pickup_area: str) -> dict:
    """EMERGENCY ONLY: immediately send the nearest free ambulance (any type) to the pickup area. Needs only the pickup area. Use when the user describes an urgent situation."""
    pos = _area(pickup_area)
    if pos is None:
        return {"error": "unknown_area", "known_areas": list(AREAS)}
    free = [a for a in FLEET if a["free"]]
    if not free:
        return {"error": "no_ambulance_available"}
    amb = min(free, key=lambda a: math.dist(a["pos"], pos))
    amb["free"] = False
    hospital = min(HOSPITALS, key=lambda h: math.dist(pos, h["pos"]))
    eta = max(2, round(math.dist(amb["pos"], pos) / 30 * 60))
    return {
        "booking_id": uuid.uuid4().hex[:8].upper(),
        "ambulance_id": amb["id"],
        "ambulance_type": amb["type"],
        "eta_minutes": eta,
        "nearest_hospital": hospital["name"],
        "note": "Dispatcher will confirm the destination hospital with the caller.",
    }

def reset_fleet():
    for a in FLEET:
        a["free"] = True