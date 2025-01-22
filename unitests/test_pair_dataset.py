import unittest
from torchvision import transforms

from config.consts import CONFIG_PATH
from src.utils.config_loader import load_config
from src.utils.load_pairs import load_pairs_from_txt_file

from data.pairs_dataset import PairsDataset


class TestPairsDataset(unittest.TestCase):
    def setUp(self):
        """Set up the LFW2 dataset with sample train pairs."""
        paths = load_config(CONFIG_PATH)["data"]

        self.train_pairs_path = paths["train_pairs_path"]
        self.data_path = paths["lfw_data_path"]

        self.train_df = load_pairs_from_txt_file(self.train_pairs_path, self.data_path)

        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])

    def test_dataset_length(self):
        """Test the length of the dataset."""
        dataset = PairsDataset(self.train_df, transform=self.transform)
        self.assertEqual(len(dataset), len(self.train_df))

    def test_getitem(self):
        """Test fetching a pair of images and label."""
        dataset = PairsDataset(self.train_df, transform=self.transform)
        (img1, img2), label = dataset[0]

        # Check the image tensor size (grayscale images)
        self.assertEqual(img1.size(), (1, 64, 64))
        self.assertEqual(img2.size(), (1, 64, 64))

        # Check the label type
        self.assertIn(label, [0, 1])  # Label should be 0 or 1


if __name__ == "__main__":
    unittest.main()
