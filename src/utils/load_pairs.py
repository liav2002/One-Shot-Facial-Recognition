import os
import pandas as pd
from typing import List

def load_pairs_from_txt_file(file_path: str, data_path: str) -> pd.DataFrame:
    """
    Load image pairs from a text file and return a DataFrame containing the pairs.

    The function processes pairs of images labeled as either the same person or different persons.
    It constructs file paths for the images based on the provided data path and the contents of the text file.

    Args:
        file_path (str): The path to the text file containing image pair information.
        data_path (str): The root directory containing the image folders.

    Returns:
        pd.DataFrame: A DataFrame with the following columns:
            - person1 (str): Name of the first person.
            - image1_path (str): File path to the first image.
            - person2 (str): Name of the second person.
            - image2_path (str): File path to the second image.
            - is_same (int): 1 if the pair is the same person, 0 otherwise.
    """
    pairs: List[dict] = []

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
