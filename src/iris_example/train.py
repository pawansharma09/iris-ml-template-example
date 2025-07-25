from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def train_model(X_train, y_train, config):
    model_params = config['model']['params']
    clf = RandomForestClassifier(**model_params)
    clf.fit(X_train, y_train)
    return clf

def save_model(model, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(model, output_path)
