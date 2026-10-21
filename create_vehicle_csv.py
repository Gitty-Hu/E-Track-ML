import pandas as pd
import os

# ---------------------------------------------------------
# 48-vehicle dataset for E-Track ride-service vehicle study
# ---------------------------------------------------------

data = [

# -------------------- BEV --------------------

["V001","Hyundai","Creta Electric 42 kWh",2026,"BEV","SUV",5,"","","2610","","","42","","420","99","","","7.4","39","","Hyundai official"],
["V002","Hyundai","Creta Electric 51.4 kWh",2026,"BEV","SUV",5,"","","2610","","","51.4","","510","126","","","7.4","39","","Hyundai official"],

["V003","Tata","Nexon EV 45 kWh",2026,"BEV","SUV",5,"","","2498","190","350","45","","489","106","","","7.2","40","","Tata official"],

["V004","Tata","Punch EV 30 kWh",2026,"BEV","SUV",5,"","","2445","195","366","30","","370","65","","","","","","Tata official"],
["V005","Tata","Punch EV 40 kWh",2026,"BEV","SUV",5,"","","2445","195","366","40","","468","95","","","","","","Tata official"],

["V006","Tata","Curvv EV 45 kWh",2026,"BEV","SUV/Coupe",5,"","","","","500","45","","430","","","", "7.2","40","","Tata official"],
["V007","Tata","Curvv EV 55 kWh",2026,"BEV","SUV/Coupe",5,"","","","","500","55","","502","123","","","7.2","40","","Tata official"],

["V008","MG","Windsor EV 38 kWh",2026,"BEV","Crossover",5,"","","2700","","","38","","332","100","","","","","","MG official"],
["V009","MG","Windsor EV PRO 52.9 kWh",2026,"BEV","Crossover",5,"","","2700","","","52.9","","449","100","","","7.4","","","MG official"],

["V010","MG","ZS EV 50.3 kWh",2026,"BEV","SUV",5,"","","2585","","","50.3","","461","130","","","7.4","54","","MG official"],

["V011","Mahindra","XUV400 34.5 kWh",2026,"BEV","SUV",5,"","","2600","","378","34.5","","375","110","","","3.3","50","","Mahindra official"],
["V012","Mahindra","XUV400 39.4 kWh",2026,"BEV","SUV",5,"","","2600","","378","39.4","","456","110","","","7.2","50","","Mahindra official"],

["V013","Mahindra","BE 6 59 kWh",2026,"BEV","SUV",5,"","","","","","59","","557","170","","","","","","Mahindra official"],
["V014","Mahindra","BE 6 79 kWh",2026,"BEV","SUV",5,"","","","","","79","","683","210","","","","","","Mahindra official"],

["V015","Mahindra","XEV 9e 59 kWh",2026,"BEV","SUV/Coupe",5,"","","","","","59","","542","170","","","","","","Mahindra official"],
["V016","Mahindra","XEV 9e 79 kWh",2026,"BEV","SUV/Coupe",5,"","","","","","79","","656","210","","","","","","Mahindra official"],

["V017","Tata","Tiago EV 19.2 kWh",2026,"BEV","Hatchback",5,"","","","","","19.2","","250","45","","","","","","Tata official"],
["V018","Tata","Tiago EV 24 kWh",2026,"BEV","Hatchback",5,"","","","","","24","","285","55","","","","","","Tata official"],

["V019","Tata","Tigor EV 26 kWh",2026,"BEV","Sedan",5,"","","","","","26","","315","55","","","","","","Tata official"],


# -------------------- CNG --------------------

["V020","Hyundai","Aura CNG",2026,"CNG","Sedan",5,"","","","","","", "37","","50.5","","","","","","Hyundai official"],

["V021","Maruti Suzuki","Dzire CNG",2026,"CNG","Sedan",5,"","","","","","", "55","","","", "31.12","","","","Maruti Suzuki official"],

["V022","Maruti Suzuki","Ertiga CNG",2026,"CNG","MPV",7,"","","2740","","","", "60","","","", "26.11","","","","Maruti Suzuki official"],

["V023","Maruti Suzuki","XL6 CNG",2026,"CNG","MPV",6,"","","2740","","","", "45","","","", "26.32","","","","Maruti Suzuki official"],

["V024","Maruti Suzuki","Brezza CNG",2026,"CNG","SUV",5,"","","2500","","","", "48","","","", "25.51","","","","Maruti Suzuki official"],

["V025","Toyota","Rumion CNG",2026,"CNG","MPV",7,"","","2740","","","", "60","","","", "26.11","","","","Toyota official"],


# -------------------- PETROL --------------------

["V026","Hyundai","Aura Petrol",2026,"Petrol","Sedan",5,"","","2450","","","", "37","","61","", "19.5","","","","Hyundai official"],

["V027","Maruti Suzuki","Dzire Petrol",2026,"Petrol","Sedan",5,"","","2450","","","", "37","","60.5","", "24.8","","","","Maruti Suzuki official"],

["V028","Honda","Amaze Petrol MT",2026,"Petrol","Sedan",5,"","","2470","","","", "40","","66","", "19.46","","","","Honda official"],

["V029","Honda","Amaze Petrol CVT",2026,"Petrol","Sedan",5,"","","2470","","","", "40","","66","", "18.65","","","","Honda official"],

["V030","Hyundai","Verna 1.5 MPi Petrol MT",2026,"Petrol","Sedan",5,"","","2670","","","", "45","","85","", "18.6","","","","Hyundai official"],

["V031","Hyundai","Verna 1.5 MPi Petrol IVT",2026,"Petrol","Sedan",5,"","","2670","","","", "45","","85","", "18.6","","","","Hyundai official"],

["V032","Hyundai","Verna 1.5 Turbo GDi MT",2026,"Petrol","Sedan",5,"","","2670","","","", "45","","117.5","", "20.0","","","","Hyundai official"],

["V033","Hyundai","Verna 1.5 Turbo GDi DCT",2026,"Petrol","Sedan",5,"","","2670","","","", "45","","117.5","", "20.0","","","","Hyundai official"],

["V034","Hyundai","Creta 1.5 MPi Petrol",2026,"Petrol","SUV",5,"","","2610","","","", "50","","85","", "17.4","","","","Hyundai official"],

["V035","Hyundai","Creta 1.5 Turbo GDi Petrol",2026,"Petrol","SUV",5,"","","2610","","","", "50","","117.5","", "18.4","","","","Hyundai official"],

["V036","Kia","Carens 1.5 Petrol",2026,"Petrol","MPV",7,"","","2780","","","", "45","","85","", "15.7","","","","Kia official"],

["V037","Kia","Carens 1.5 Turbo Petrol MT",2026,"Petrol","MPV",7,"","","2780","","","", "45","","118","", "16.5","","","","Kia official"],

["V038","Kia","Carens 1.5 Turbo Petrol DCT",2026,"Petrol","MPV",7,"","","2780","","","", "45","","118","", "15.8","","","","Kia official"],

["V039","Kia","Carens 1.5 Diesel MT",2026,"Diesel","MPV",7,"","","2780","","","", "45","","85","", "21.3","","","","Kia official"],

["V040","Kia","Carens 1.5 Diesel AT",2026,"Diesel","MPV",7,"","","2780","","","", "45","","85","", "18.4","","","","Kia official"],


# -------------------- HYBRID --------------------

["V041","Toyota","Urban Cruiser Hyryder Strong Hybrid",2026,"Hybrid","SUV",5,"","","2600","","","", "45","","85","", "27.97","","","","Toyota official"],

["V042","Maruti Suzuki","Grand Vitara Strong Hybrid",2026,"Hybrid","SUV",5,"","","2600","","","", "45","","85","", "27.97","","","","Maruti Suzuki official"],

["V043","Honda","City e:HEV",2026,"Hybrid","Sedan",5,"","","2600","","","", "40","","93","", "26.5","","","","Honda official"],

["V044","Toyota","Innova Hycross Hybrid",2026,"Hybrid","MPV",7,"","","2850","","","", "52","","139","", "23.24","","","","Toyota official"],


# -------------------- ADDITIONAL PETROL --------------------

["V045","Toyota","Innova Hycross Petrol",2026,"Petrol","MPV",8,"","","2850","","","", "52","","126","", "16.13","","","","Toyota official"],

["V046","Maruti Suzuki","Ertiga Petrol",2026,"Petrol","MPV",7,"","","2740","","","", "45","","75.8","", "20.51","","","","Maruti Suzuki official"],

["V047","Maruti Suzuki","XL6 Petrol",2026,"Petrol","MPV",6,"","","2740","","","", "45","","75.8","", "20.97","","","","Maruti Suzuki official"],

["V048","Maruti Suzuki","Brezza Petrol",2026,"Petrol","SUV",5,"","","2500","","","", "48","","75.8","", "17.8","","","","Maruti Suzuki official"]

]

# ---------------------------------------------------------
# Column names
# ---------------------------------------------------------

columns = [
    "vehicle_id",
    "manufacturer",
    "model",
    "model_year",
    "powertrain",
    "vehicle_type",
    "seats",
    "kerb_weight_kg",
    "payload_kg",
    "wheelbase_mm",
    "ground_clearance_mm",
    "luggage_volume_l",
    "battery_capacity_kwh",
    "fuel_tank_l",
    "certified_range_km",
    "power_kw",
    "energy_consumption_kwh_100km",
    "fuel_consumption_l_100km",
    "charging_power_kw",
    "charging_time_min",
    "purchase_price_inr",
    "data_source"
]

# ---------------------------------------------------------
# Create DataFrame
# ---------------------------------------------------------

df = pd.DataFrame(data, columns=columns)

# Convert empty strings to proper missing values
df = df.replace("", pd.NA)

# ---------------------------------------------------------
# Save to the required location
# ---------------------------------------------------------

output_path = os.path.join(
    "data",
    "raw",
    "vehicles_raw.csv"
)

df.to_csv(output_path, index=False)

# ---------------------------------------------------------
# Verification
# ---------------------------------------------------------

print("\nVehicle CSV created successfully!")
print(f"Location: {output_path}")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

print("\nVehicle IDs:")
print(df["vehicle_id"].tolist())

print("\nShape:")
print(df.shape)