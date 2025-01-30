from src.scripts.train import train_model
from src.scripts.test import test_model
from src.scripts.predict import predict


def main():
    # test_model(run_name="Siamese_Test_with_Best_Model")
    predict("run_001", "Ahmad_Masood1", "./data/lfw2/Ahmad_Masood/Ahmad_Masood_0001.jpg", "Ahmad_Masood2", "./data/lfw2/Ahmad_Masood/Ahmad_Masood_0002.jpg")


if __name__ == "__main__":
    main()
