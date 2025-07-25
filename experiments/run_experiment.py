import yaml
import sys
import os

sys.path.append(os.path.abspath("../src"))

from iris_example.data_loader import load_data
from iris_example.train import train_model, save_model
from iris_example.evaluate import evaluate_model

def main():
    with open("configs/train_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    X_train, X_test, y_train, y_test = load_data(
        test_size=config['test_size'],
        random_seed=config['random_seed']
    )

    model = train_model(X_train, y_train, config)
    evaluate_model(model, X_test, y_test)
    save_model(model, config['output_path'])

if __name__ == "__main__":
    main()
