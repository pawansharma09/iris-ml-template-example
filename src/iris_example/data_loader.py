from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def load_data(test_size=0.2, random_seed=42):
    iris = load_iris()
    X, y = iris.data, iris.target
    return train_test_split(X, y, test_size=test_size, random_state=random_seed)
