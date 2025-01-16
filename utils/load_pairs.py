import os
import pandas as pd


def load_pairs_from_txt_file(file_path, data_path):
    pairs = []

    with open(file_path, 'r') as file:
        lines = file.readlines()
        same_pairs_count = int(lines[0].strip())

        # Process same-person pairs
        for line in lines[1:same_pairs_count + 1]:
            parts = line.strip().split()
            person_name = parts[0]
            image1_path = os.path.join(data_path, person_name, f"{person_name}_{int(parts[1]):04d}.jpg")
            image2_path = os.path.join(data_path, person_name, f"{person_name}_{int(parts[2]):04d}.jpg")
            pairs.append({
                "person1": person_name,
                "image1_path": image1_path,
                "person2": person_name,
                "image2_path": image2_path,
                "is_same": 1
            })

        # Process different-person pairs
        for line in lines[same_pairs_count + 1:]:
            parts = line.strip().split()
            person1_name = parts[0]
            image1_path = os.path.join(data_path, person1_name, f"{person1_name}_{int(parts[1]):04d}.jpg")
            person2_name = parts[2]
            image2_path = os.path.join(data_path, person2_name, f"{person2_name}_{int(parts[3]):04d}.jpg")
            pairs.append({
                "person1": person1_name,
                "image1_path": image1_path,
                "person2": person2_name,
                "image2_path": image2_path,
                "is_same": 0
            })

    return pd.DataFrame(pairs)
