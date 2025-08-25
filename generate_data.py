import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import csv

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Asset Categories and Types
WEAPONS = [
    "Glock 19",
    "Sig Sauer P226",
    "Walther P99",
    "Revolver .38 Special",
    "M4 Carbine",
    "M16A4",
    "HK416",
    "Remington 870",
    "Benelli M4",
    "MP5",
    "Uzi",
    "Taser",
    "Baton",
]

ICT_ASSETS = [
    "Cisco ISR Router",
    "Huawei AR Router",
    "Cisco Catalyst Switch",
    "HP Aruba Switch",
    "Fortinet Firewall",
    "Palo Alto Firewall",
    "Windows Server 2019",
    "Linux RHEL Server",
    "VMware ESXi",
    "HP Laptop",
    "iPad Tablet",
    "Canon Printer",
    "Malware Threat Management",
    "DNS Server",
    "NAS Storage",
    "SAN Storage",
]

VEHICLES = [
    "Proton Wira",
    "Proton X70",
    "Honda Civic",
    "Toyota Hilux",
    "Honda CBX Motorcycle",
    "Yamaha R25 Motorcycle",
    "APC",
    "Riot Control Vehicle",
    "Transport Truck",
    "Logistics Lorry",
    "Prison Transport Bus",
    "Staff Bus",
]

DEVICES = [
    "Axon Body 3",
    "Motorola VB400",
    "Dahua CCTV",
    "Hikvision CCTV",
    "DJI Mavic Drone",
    "Parrot Anafi Drone",
    "Motorola APX Radio",
    "Fingerprint Scanner",
    "Facial Recognition Terminal",
]

# Malaysian States and Cities
STATES_CITIES = {
    "Selangor": ["Shah Alam", "Subang Jaya", "Klang", "Petaling Jaya"],
    "Johor": ["Johor Bahru", "Skudai", "Batu Pahat", "Muar"],
    "Penang": ["George Town", "Butterworth", "Bukit Mertajam", "Balik Pulau"],
    "Sabah": ["Kota Kinabalu", "Sandakan", "Tawau", "Lahad Datu"],
    "Sarawak": ["Kuching", "Miri", "Sibu", "Bintulu"],
    "Kuala Lumpur": ["Kuala Lumpur", "Cheras", "Kepong", "Bangsar"],
    "Kedah": ["Alor Setar", "Sungai Petani", "Kulim", "Langkawi"],
    "Pahang": ["Kuantan", "Temerloh", "Bentong", "Jerantut"],
    "Perak": ["Ipoh", "Taiping", "Sitiawan", "Kampar"],
    "Negeri Sembilan": ["Seremban", "Nilai", "Port Dickson", "Bahau"],
    "Melaka": ["Melaka City", "Jasin", "Alor Gajah", "Ayer Keroh"],
    "Terengganu": ["Kuala Terengganu", "Kemaman", "Dungun", "Chukai"],
    "Kelantan": ["Kota Bharu", "Pasir Mas", "Tanah Merah", "Machang"],
    "Putrajaya": ["Putrajaya", "Presint 1", "Presint 8", "Presint 15"],
}

# Officer names (Malay names)
MALAY_NAMES = [
    "Ahmad Faiz",
    "Noraini",
    "Syafiq",
    "Aisyah",
    "Razak",
    "Siti Nurhaliza",
    "Mohd Azlan",
    "Fatimah",
    "Hafiz",
    "Zarina",
    "Abdul Rahman",
    "Nadia",
    "Ismail",
    "Rohani",
    "Farid",
    "Mariam",
    "Azmi",
    "Nurul",
    "Hakim",
    "Sarah",
    "Baharuddin",
    "Aminah",
    "Zulkifli",
    "Haslinda",
    "Rashid",
    "Normah",
    "Kamarul",
    "Sharifah",
    "Mansor",
    "Khadijah",
    "Azhar",
    "Rosnah",
    "Ibrahim",
    "Suraya",
    "Kamal",
    "Fauziah",
    "Yusof",
    "Zaharah",
    "Omar",
    "Halimah",
    "Hamid",
    "Ramlah",
    "Hassan",
    "Zaitun",
]

# Conditions and remarks
CONDITIONS = ["Good", "Needs Action", "Damaged"]
REMARKS = [
    "Requires software upgrade",
    "Engine overheating",
    "Camera lens scratched",
    "Battery degradation",
    "Scheduled for maintenance",
    "Performance optimal",
    "Minor wear and tear",
    "Needs calibration",
    "Software update pending",
    "Hardware replacement needed",
    "Regular maintenance completed",
    "No issues",
    "Requires cleaning",
    "Battery replacement needed",
    "Firmware outdated",
    "Network connectivity issues",
    "Storage capacity low",
    "Regular inspection due",
]

# Weapon calibers
CALIBERS = [
    "9mm",
    ".22 LR",
    ".38 Special",
    ".357 Magnum",
    "5.56mm",
    "7.62mm",
    ".45 ACP",
]

# Fuel types
FUEL_TYPES = ["Petrol", "Diesel", "Electric", "Hybrid"]

# Units
UNITS = [
    "Traffic Police",
    "Criminal Investigation",
    "Special Branch",
    "Patrol Unit",
    "Cybercrime Unit",
    "Narcotics Division",
    "Anti-Vice Unit",
    "Mobile Patrol Force",
]


def generate_asset_id():
    """Generate a unique asset ID"""
    prefix = random.choice(["PDM", "AST", "EQP"])
    number = random.randint(100000, 999999)
    return f"{prefix}{number}"


def generate_serial_number():
    """Generate a realistic serial number"""
    letters = "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=2))
    numbers = "".join(random.choices("0123456789", k=6))
    return f"{letters}{numbers}"


def generate_license_plate():
    """Generate Malaysian license plate"""
    prefixes = [
        "WA",
        "WB",
        "WC",
        "KL",
        "SG",
        "JH",
        "PN",
        "KD",
        "PH",
        "NS",
        "ML",
        "TR",
        "KB",
    ]
    prefix = random.choice(prefixes)
    number = random.randint(1000, 9999)
    suffix = random.choice(["A", "B", "C", "D", "E", "F", "G", "H", "J", "K"])
    return f"{prefix} {number} {suffix}"


def generate_ip_address():
    """Generate a realistic internal IP address"""
    subnets = ["192.168.", "10.0.", "172.16."]
    subnet = random.choice(subnets)
    octets = [str(random.randint(1, 254)), str(random.randint(1, 254))]
    return subnet + ".".join(octets)


def generate_random_date(start_year=2015, end_year=2025):
    """Generate a random date between start_year and end_year"""
    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 12, 31)
    time_between = end_date - start_date
    days_between = time_between.days
    random_days = random.randrange(days_between)
    return start_date + timedelta(days=random_days)


def generate_maintenance_dates(purchase_date):
    """Generate realistic maintenance dates"""
    last_maintenance = purchase_date + timedelta(days=random.randint(30, 365))
    if last_maintenance > datetime.now():
        last_maintenance = datetime.now() - timedelta(days=random.randint(1, 30))

    next_maintenance = last_maintenance + timedelta(days=random.randint(90, 365))
    return last_maintenance, next_maintenance


def get_state_city():
    """Get a random state and city combination"""
    state = random.choice(list(STATES_CITIES.keys()))
    city = random.choice(STATES_CITIES[state])
    return state, city


def generate_weapon_data():
    """Generate weapon-specific data with realistic distribution"""
    # Weighted selection for more realistic weapon distribution
    weapon_weights = {
        "Glock 19": 0.25,  # Most common sidearm
        "Sig Sauer P226": 0.15,
        "Walther P99": 0.10,
        "Revolver .38 Special": 0.08,
        "M4 Carbine": 0.12,  # Common patrol rifle
        "M16A4": 0.08,
        "HK416": 0.05,  # Specialized units
        "Remington 870": 0.07,  # Shotgun
        "Benelli M4": 0.03,
        "MP5": 0.04,  # SMG for special units
        "Uzi": 0.02,
        "Taser": 0.06,  # Non-lethal
        "Baton": 0.10,  # Very common
    }

    # Select weapon type based on weights
    weapons = list(weapon_weights.keys())
    weights = list(weapon_weights.values())
    asset_type = random.choices(weapons, weights=weights)[0]

    state, city = get_state_city()
    purchase_date = generate_random_date()
    last_maint, next_maint = generate_maintenance_dates(purchase_date)

    return {
        "asset_id": generate_asset_id(),
        "category": "Weapons",
        "type": asset_type,
        "state": state,
        "city": city,
        "condition": random.choice(CONDITIONS),
        "purchase_date": purchase_date.strftime("%Y-%m-%d"),
        "last_maintenance_date": last_maint.strftime("%Y-%m-%d"),
        "next_maintenance_due": next_maint.strftime("%Y-%m-%d"),
        "usage_hours": random.randint(100, 5000),
        "assigned_officer": random.choice(MALAY_NAMES),
        "remarks": random.choice(REMARKS),
        "serial_number": generate_serial_number(),
        "caliber": random.choice(CALIBERS),
        "ammunition_count": random.randint(0, 200),
        "assigned_unit": random.choice(UNITS),
        # Null values for other categories
        "ip_address": None,
        "firmware_version": None,
        "os_version": None,
        "vendor": None,
        "warranty_expiry": None,
        "license_plate": None,
        "fuel_type": None,
        "assigned_station": None,
        "mileage_km": None,
        "battery_health": None,
        "connectivity": None,
        "storage_capacity": None,
    }


def generate_ict_data():
    """Generate ICT asset-specific data with realistic distribution"""
    # Weighted selection for more realistic ICT asset distribution
    ict_weights = {
        "HP Laptop": 0.30,  # Most common
        "iPad Tablet": 0.15,
        "Canon Printer": 0.12,
        "Windows Server 2019": 0.08,
        "Linux RHEL Server": 0.05,
        "Cisco ISR Router": 0.07,
        "Huawei AR Router": 0.04,
        "Cisco Catalyst Switch": 0.06,
        "HP Aruba Switch": 0.03,
        "Fortinet Firewall": 0.04,
        "Palo Alto Firewall": 0.02,
        "VMware ESXi": 0.03,
        "DNS Server": 0.02,
        "NAS Storage": 0.02,
        "SAN Storage": 0.01,
        "Malware Threat Management": 0.02,
    }

    # Select ICT asset type based on weights
    ict_assets = list(ict_weights.keys())
    weights = list(ict_weights.values())
    asset_type = random.choices(ict_assets, weights=weights)[0]

    state, city = get_state_city()
    purchase_date = generate_random_date()
    last_maint, next_maint = generate_maintenance_dates(purchase_date)

    vendors = {
        "Cisco ISR Router": "Cisco",
        "Huawei AR Router": "Huawei",
        "Cisco Catalyst Switch": "Cisco",
        "HP Aruba Switch": "HP",
        "Fortinet Firewall": "Fortinet",
        "Palo Alto Firewall": "Palo Alto",
        "Windows Server 2019": "Microsoft",
        "Linux RHEL Server": "Red Hat",
        "VMware ESXi": "VMware",
        "HP Laptop": "HP",
        "iPad Tablet": "Apple",
        "Canon Printer": "Canon",
        "Malware Threat Management": "Symantec",
        "DNS Server": "Microsoft",
        "NAS Storage": "Synology",
        "SAN Storage": "Dell EMC",
    }

    warranty_date = purchase_date + timedelta(
        days=random.randint(365, 1825)
    )  # 1-5 years

    return {
        "asset_id": generate_asset_id(),
        "category": "ICT Assets",
        "type": asset_type,
        "state": state,
        "city": city,
        "condition": random.choice(CONDITIONS),
        "purchase_date": purchase_date.strftime("%Y-%m-%d"),
        "last_maintenance_date": last_maint.strftime("%Y-%m-%d"),
        "next_maintenance_due": next_maint.strftime("%Y-%m-%d"),
        "usage_hours": random.randint(500, 8760),  # Up to 1 year of hours
        "assigned_officer": random.choice(MALAY_NAMES),
        "remarks": random.choice(REMARKS),
        "ip_address": generate_ip_address(),
        "firmware_version": f"v{random.randint(1,5)}.{random.randint(0,9)}.{random.randint(0,9)}",
        "os_version": f"v{random.randint(8,12)}.{random.randint(0,5)}",
        "vendor": vendors.get(asset_type, "Unknown"),
        "warranty_expiry": warranty_date.strftime("%Y-%m-%d"),
        # Null values for other categories
        "serial_number": None,
        "caliber": None,
        "ammunition_count": None,
        "assigned_unit": None,
        "license_plate": None,
        "fuel_type": None,
        "assigned_station": None,
        "mileage_km": None,
        "battery_health": None,
        "connectivity": None,
        "storage_capacity": None,
    }


def generate_vehicle_data():
    """Generate vehicle-specific data with realistic distribution"""
    # Weighted selection for more realistic vehicle distribution
    vehicle_weights = {
        "Proton Wira": 0.20,  # Common patrol car
        "Proton X70": 0.15,  # Newer patrol car
        "Honda Civic": 0.12,  # Patrol car
        "Toyota Hilux": 0.15,  # Pickup truck for terrain
        "Honda CBX Motorcycle": 0.18,  # Most common patrol bike
        "Yamaha R25 Motorcycle": 0.10,  # Patrol bike
        "Transport Truck": 0.04,  # Logistics
        "Logistics Lorry": 0.03,
        "Staff Bus": 0.02,  # Personnel transport
        "Prison Transport Bus": 0.01,  # Specialized
        "APC": 0.001,  # Very rare
        "Riot Control Vehicle": 0.001,  # Very rare
    }

    # Select vehicle type based on weights
    vehicles = list(vehicle_weights.keys())
    weights = list(vehicle_weights.values())
    asset_type = random.choices(vehicles, weights=weights)[0]

    state, city = get_state_city()
    purchase_date = generate_random_date()
    last_maint, next_maint = generate_maintenance_dates(purchase_date)

    stations = [f"{city} Police Station", f"{city} HQ", f"{city} District Office"]

    return {
        "asset_id": generate_asset_id(),
        "category": "Vehicles",
        "type": asset_type,
        "state": state,
        "city": city,
        "condition": random.choice(CONDITIONS),
        "purchase_date": purchase_date.strftime("%Y-%m-%d"),
        "last_maintenance_date": last_maint.strftime("%Y-%m-%d"),
        "next_maintenance_due": next_maint.strftime("%Y-%m-%d"),
        "usage_hours": None,  # Vehicles use mileage instead
        "assigned_officer": random.choice(MALAY_NAMES),
        "remarks": random.choice(REMARKS),
        "license_plate": generate_license_plate(),
        "fuel_type": random.choice(FUEL_TYPES),
        "assigned_station": random.choice(stations),
        "mileage_km": random.randint(1000, 200000),
        # Null values for other categories
        "serial_number": None,
        "caliber": None,
        "ammunition_count": None,
        "assigned_unit": None,
        "ip_address": None,
        "firmware_version": None,
        "os_version": None,
        "vendor": None,
        "warranty_expiry": None,
        "battery_health": None,
        "connectivity": None,
        "storage_capacity": None,
    }


def generate_device_data():
    """Generate device-specific data with realistic distribution"""
    # Weighted selection for more realistic device distribution
    device_weights = {
        "Axon Body 3": 0.25,  # Body cameras very common
        "Motorola VB400": 0.20,  # Another body cam
        "Motorola APX Radio": 0.20,  # Communication radios very common
        "Dahua CCTV": 0.15,  # CCTV cameras
        "Hikvision CCTV": 0.12,  # CCTV cameras
        "Fingerprint Scanner": 0.04,  # Specialized equipment
        "Facial Recognition Terminal": 0.02,
        "DJI Mavic Drone": 0.015,  # Drones less common
        "Parrot Anafi Drone": 0.005,  # Specialized drones
    }

    # Select device type based on weights
    devices = list(device_weights.keys())
    weights = list(device_weights.values())
    asset_type = random.choices(devices, weights=weights)[0]

    state, city = get_state_city()
    purchase_date = generate_random_date()
    last_maint, next_maint = generate_maintenance_dates(purchase_date)

    connectivity_types = ["WiFi", "Bluetooth", "4G", "5G", "Ethernet", "USB"]
    storage_capacities = ["16GB", "32GB", "64GB", "128GB", "256GB", "512GB", "1TB"]

    return {
        "asset_id": generate_asset_id(),
        "category": "Devices",
        "type": asset_type,
        "state": state,
        "city": city,
        "condition": random.choice(CONDITIONS),
        "purchase_date": purchase_date.strftime("%Y-%m-%d"),
        "last_maintenance_date": last_maint.strftime("%Y-%m-%d"),
        "next_maintenance_due": next_maint.strftime("%Y-%m-%d"),
        "usage_hours": random.randint(100, 3000),
        "assigned_officer": random.choice(MALAY_NAMES),
        "remarks": random.choice(REMARKS),
        "battery_health": f"{random.randint(60, 100)}%",
        "firmware_version": f"v{random.randint(1,10)}.{random.randint(0,9)}",
        "connectivity": random.choice(connectivity_types),
        "storage_capacity": random.choice(storage_capacities),
        # Null values for other categories
        "serial_number": None,
        "caliber": None,
        "ammunition_count": None,
        "assigned_unit": None,
        "ip_address": None,
        "os_version": None,
        "vendor": None,
        "warranty_expiry": None,
        "license_plate": None,
        "fuel_type": None,
        "assigned_station": None,
        "mileage_km": None,
    }


def generate_dataset(total_rows=20000):
    """Generate the complete dataset with realistic distribution"""
    print(f"Generating {total_rows} rows of synthetic PDRM asset data...")

    data = []

    # Realistic distribution based on typical police force asset allocation
    # Weapons: 35% (most common - every officer needs weapons)
    # Vehicles: 30% (patrol cars, motorcycles, transport)
    # ICT Assets: 25% (computers, servers, network equipment)
    # Devices: 10% (specialized equipment like cameras, drones)

    category_weights = {
        "Weapons": 0.35,
        "Vehicles": 0.30,
        "ICT Assets": 0.25,
        "Devices": 0.10,
    }

    # Calculate realistic distribution
    for category, weight in category_weights.items():
        category_count = int(total_rows * weight)
        print(f"Generating {category_count} {category} records ({weight*100:.1f}%)...")

        for _ in range(category_count):
            if category == "Weapons":
                data.append(generate_weapon_data())
            elif category == "ICT Assets":
                data.append(generate_ict_data())
            elif category == "Vehicles":
                data.append(generate_vehicle_data())
            elif category == "Devices":
                data.append(generate_device_data())

    # Add remaining records to reach exact total (distribute randomly)
    remaining = total_rows - len(data)
    if remaining > 0:
        print(f"Adding {remaining} additional records...")
        categories = list(category_weights.keys())
        for _ in range(remaining):
            category = random.choice(categories)
            if category == "Weapons":
                data.append(generate_weapon_data())
            elif category == "ICT Assets":
                data.append(generate_ict_data())
            elif category == "Vehicles":
                data.append(generate_vehicle_data())
            elif category == "Devices":
                data.append(generate_device_data())

    # Shuffle the data to mix categories
    random.shuffle(data)

    print(f"Total records generated: {len(data)}")
    return data


def main():
    """Main function to generate and save the dataset"""
    print("Starting PDRM Asset & Facility Monitoring Dataset Generation")
    print("=" * 60)

    # Generate the dataset with 20,000 records
    dataset = generate_dataset(20000)

    # Convert to DataFrame
    print("Converting to DataFrame...")
    df = pd.DataFrame(dataset)

    # Reorder columns for better readability
    column_order = [
        "asset_id",
        "category",
        "type",
        "state",
        "city",
        "condition",
        "purchase_date",
        "last_maintenance_date",
        "next_maintenance_due",
        "usage_hours",
        "assigned_officer",
        "remarks",
        # Weapon-specific
        "serial_number",
        "caliber",
        "ammunition_count",
        "assigned_unit",
        # ICT-specific
        "ip_address",
        "firmware_version",
        "os_version",
        "vendor",
        "warranty_expiry",
        # Vehicle-specific
        "license_plate",
        "fuel_type",
        "assigned_station",
        "mileage_km",
        # Device-specific
        "battery_health",
        "connectivity",
        "storage_capacity",
    ]

    df = df[column_order]

    # Save to CSV
    output_file = "assets_data.csv"
    print(f"Saving dataset to {output_file}...")
    df.to_csv(output_file, index=False)

    # Display summary statistics
    print("\nDataset Summary:")
    print("=" * 40)
    print(f"Total records: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    print("\nRecords by category:")
    print(df["category"].value_counts())
    print("\nRecords by condition:")
    print(df["condition"].value_counts())
    print("\nRecords by state:")
    print(df["state"].value_counts())

    print(f"\nDataset successfully saved as '{output_file}'")
    print("Generation complete!")


if __name__ == "__main__":
    main()
