import json
import csv
import math

def convert_json_to_csv(json_filepath, csv_filepath):
    # Load JSON data
    with open(json_filepath, 'r') as file:
        data = json.load(file)
    
    # Extract time keys and sort them
    time_keys = sorted(data["time_s"].keys(), key=float)
    
    # Prepare CSV headers
    boats = list(data["time_s"][time_keys[0]].keys())
    headers = ["time"]
    for idx, boat in enumerate(boats):
        headers.extend([f"V{idx+1}_y", f"V{idx+1}_x", f"V{idx+1}_h"])
    
    # Write to CSV
    with open(csv_filepath, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        
        for time in time_keys:
            row = [time]
            for boat in boats:
                position = data["time_s"][time][boat]["center_position_m"]
                position[0] -= 130  # Undo transformation on x
                position[1] *= -1  # Undo transformation on y
                position[1] += 15

                heading_rad = data["time_s"][time][boat]["heading_rad"]
                heading = 180.0 * heading_rad / math.pi  + 90.0
                row.extend([position[0], position[1], heading])
            writer.writerow(row)

# Example usage
convert_json_to_csv("extracted_data/boats/dynamics_simon.json", "9.csv")
