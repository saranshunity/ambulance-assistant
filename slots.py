from typing import Optional, Literal
from pydantic import BaseModel


class BookingSlots(BaseModel):
    patient_name: Optional[str] = None
    phone: Optional[str] = None
    pickup_location: Optional[str] = None
    hospital: Optional[str] = None
    ambulance_type: Optional[Literal["basic", "oxygen", "icu"]] = None
    is_emergency: bool = False