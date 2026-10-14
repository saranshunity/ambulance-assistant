from tools import find_hospitals, estimate_fare, create_booking

print(find_hospitals("Green Park"))
print(find_hospitals("Mars"))
print(estimate_fare("Green Park", "Demo City Hospital", "icu"))
print(create_booking("Rahul", "0000000000", "Green Park", "Demo City Hospital", "icu"))