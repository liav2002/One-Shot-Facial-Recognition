import unittest
import torch
from model.siamese_network import SiameseNetwork


class TestSiameseNetwork(unittest.TestCase):
    def setUp(self):
        """Initialize the Siamese Network and create dummy inputs."""
        self.model = SiameseNetwork()
        self.x1 = torch.randn(2, 1, 105, 105)
        self.x2 = torch.randn(2, 1, 105, 105)

    def test_forward_pass(self):
        """Test the forward pass of the Siamese Network."""
        output = self.model(self.x1, self.x2)
        self.assertEqual(output.shape, (2, 1), "Output shape should be [B, 1]")
        self.assertTrue(torch.all(output >= 0) and torch.all(output <= 1), "Output values should be between 0 and 1")

    def test_shared_weights(self):
        """Test if the twin networks share weights."""
        cnn_params = list(self.model.cnn.parameters())
        self.assertTrue(all(torch.equal(cnn_params[i], cnn_params[i]) for i in range(len(cnn_params))),
                        "Weights in twin networks should be shared")

    def test_large_batch(self):
        """Test the model with a larger batch size."""
        x1_large = torch.randn(32, 1, 105, 105)
        x2_large = torch.randn(32, 1, 105, 105)
        output = self.model(x1_large, x2_large)
        self.assertEqual(output.shape, (32, 1), "Output shape should match batch size [B, 1]")


if __name__ == "__main__":
    unittest.main()
