import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from typing import Callable, Optional, Tuple


class PairsDataset(Dataset):
    def __init__(self, pairs_df: pd.DataFrame, transform: Optional[Callable] = None):
        """
        Args:
            pairs_df (pd.DataFrame): DataFrame containing pairs data.
                Columns: ['person1', 'image1_path', 'person2', 'image2_path', 'is_same'].
            transform (Callable, optional): Transformations to apply to the images.
        """
        self.pairs_df = pairs_df
        self.transform = transform

    def __len__(self) -> int:
        """Returns the total number of pairs."""
        return len(self.pairs_df)

    def __getitem__(self, index: int) -> Tuple[Tuple[torch.Tensor, torch.Tensor], int]:
        """
        Fetches a pair of images and their corresponding label.

        Args:
            index (int): Index of the pair.

        Returns:
            Tuple[Tuple[torch.Tensor, torch.Tensor], int]:
                - Two image tensors (transformed if applicable).
                - A label (1 if same person, 0 if different people).
        """
        row = self.pairs_df.iloc[index]

        image1 = Image.open(row['image1_path']).convert("L")
        image2 = Image.open(row['image2_path']).convert("L")

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        label = row['is_same']
        return (image1, image2), label
