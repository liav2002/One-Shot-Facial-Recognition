import torch
import random
import pandas as pd
import networkx as nx
from typing import Tuple
from tabulate import tabulate
from sklearn.model_selection import train_test_split

from src.utils.logger import get_logger


def split_by_unique_persons(
        full_df: pd.DataFrame,
        val_split: float,
        shuffle: bool,
        random_seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits a dataframe of pairs of people into training and validation sets based on unique people,
    and calculates the percentage of pairs assigned to each set. The function also logs the split
    statistics in a table format.

    Args:
        full_df (pd.DataFrame): The full dataframe containing pairs of people. It must contain
            two columns 'person1' and 'person2' that represent the pairs of people.
        val_split (float): The fraction of unique people to be assigned to the validation set.
            The remaining will be assigned to the training set. Should be a value between 0 and 1.
        shuffle (bool): Whether to shuffle the data before splitting.
        random_seed (int): The random seed used to ensure reproducibility of the split.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing two dataframes:
            - The training dataframe (`train_df`), containing the pairs assigned to the training set.
            - The validation dataframe (`val_df`), containing the pairs assigned to the validation set.
    """
    logger = get_logger()

    logger.log_message(
        f"Splitting Peoples by:\nTrain People - {(1 - val_split) * 100}%\nVal People - {val_split * 100}%\n")

    unique_people = pd.concat([full_df['person1'], full_df['person2']]).unique()

    train_people, val_people = train_test_split(
        unique_people, test_size=val_split,
        shuffle=shuffle,
        random_state=random_seed
    )

    # Assign pairs to train and val
    train_df = full_df[
        full_df['person1'].isin(train_people) & full_df['person2'].isin(train_people)
        ]

    val_df = full_df[
        full_df['person1'].isin(val_people) & full_df['person2'].isin(val_people)
        ]

    # Identify dropped pairs
    train_mask = full_df['person1'].isin(train_people) & full_df['person2'].isin(train_people)
    val_mask = full_df['person1'].isin(val_people) & full_df['person2'].isin(val_people)
    dropped_pairs_df = full_df[~(train_mask | val_mask)]

    # Calculate percentages
    total_pairs_len = len(full_df)
    train_pairs_len = len(train_df)
    val_pairs_len = len(val_df)
    dropped_pairs_len = len(dropped_pairs_df)

    train_pairs_percent = (train_pairs_len / total_pairs_len) * 100
    val_pairs_percent = (val_pairs_len / total_pairs_len) * 100
    dropped_pairs_percent = (dropped_pairs_len / total_pairs_len) * 100

    # Prepare data for tabulate
    split_results = [
        ["Train People", len(train_people)],
        ["Train Pairs", train_pairs_len, f"({train_pairs_percent:.2f}%)"],
        ["Validation People", len(val_people)],
        ["Validation Pairs", val_pairs_len, f"({val_pairs_percent:.2f}%)"],
        ["Dropped Pairs", dropped_pairs_len, f"({dropped_pairs_percent:.2f}%)"]
    ]

    # Log the results with tabulate
    logger.log_message("Split Results:")
    logger.log_message("\n" + tabulate(split_results, headers=["Metric", "Count", "Percentage"], tablefmt="grid"))

    return train_df, val_df


def split_pairs_by_connected_components(
        full_df: pd.DataFrame,
        val_split: float,
        random_seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits a dataframe of pairs of people into training and validation sets based on connected
    components of a graph formed by pairs of people. Each connected component will be fully assigned
    to either the training or validation set.

    Args:
        full_df (pd.DataFrame): The full dataframe containing pairs of people. It must contain two
            columns 'person1' and 'person2' that represent the pairs of people.
        val_split (float): The fraction of connected components to be assigned to the validation set.
            The remaining will be assigned to the training set. Should be a value between 0 and 1.
        random_seed (int): The random seed used to ensure reproducibility of the split.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing two dataframes:
            - The training dataframe (`train_df`), containing the pairs assigned to the training set.
            - The validation dataframe (`val_df`), containing the pairs assigned to the validation set.
    """
    logger = get_logger()
    random.seed(random_seed)

    # Create graph where nodes are people and edges represent connections between people
    graph = nx.Graph()

    # Add edges for all pairs in the dataframe
    for _, row in full_df.iterrows():
        graph.add_edge(row['person1'], row['person2'])

    # Get connected components (groups of connected people)
    connected_components = list(nx.connected_components(graph))

    # Shuffle the components to ensure randomness
    random.shuffle(connected_components)

    # Split connected components into train and validation sets
    num_val_components = int(len(connected_components) * val_split)
    val_components = connected_components[:num_val_components]
    train_components = connected_components[num_val_components:]

    # Step 2: Create train_df and val_df based on the components
    train_people = set()
    val_people = set()
    dropped_pairs = set()

    # Add persons from each component to train or validation
    for component in train_components:
        train_people.update(component)

    for component in val_components:
        val_people.update(component)

    # Identify dropped pairs (those that are in components not assigned to train or val)
    for _, row in full_df.iterrows():
        person1 = row['person1']
        person2 = row['person2']

        if person1 not in train_people and person1 not in val_people:
            dropped_pairs.add((person1, person2))
        if person2 not in train_people and person2 not in val_people:
            dropped_pairs.add((person1, person2))

    # Create train and validation DataFrames
    train_df = full_df[full_df['person1'].isin(train_people) & full_df['person2'].isin(train_people)]
    val_df = full_df[full_df['person1'].isin(val_people) | full_df['person2'].isin(val_people)]

    # Logging to show the split
    total_pairs_len = len(full_df)
    train_pairs_len = len(train_df)
    val_pairs_len = len(val_df)
    dropped_pairs_len = len(dropped_pairs)

    train_pairs_percent = (train_pairs_len / total_pairs_len) * 100
    val_pairs_percent = (val_pairs_len / total_pairs_len) * 100
    dropped_pairs_percent = (dropped_pairs_len / total_pairs_len) * 100

    # Prepare data for tabulate
    split_results = [
        ["Train People", len(train_people)],
        ["Train Pairs", train_pairs_len, f"({train_pairs_percent:.2f}%)"],
        ["Validation People", len(val_people)],
        ["Validation Pairs", val_pairs_len, f"({val_pairs_percent:.2f}%)"],
        ["Dropped Pairs", dropped_pairs_len, f"({dropped_pairs_percent:.2f}%)"]
    ]

    # Print table using tabulate
    logger.log_message("Split Results:")
    logger.log_message("\n" + tabulate(split_results, headers=["Metric", "Count", "Percentage"], tablefmt="grid"))

    return train_df, val_df
