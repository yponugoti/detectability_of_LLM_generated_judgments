import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

datasets = {
     "Helpsteer2": ["helpfulness", "correctness", "coherence", "complexity", "verbosity"],
     "Helpsteer3": ["score"],
     "Neurips":    ["rating", "confidence", "soundness", "presentation", "contribution"],
     "ANTIQUE":    ["ranking"]
}

def load_data(dataset, split, groupSize):
    filepath = f"data/dataset_detection/gpt-4o-2024-08-06_{dataset.lower()}_{split}_{groupSize}_grouped/gpt-4o-2024-08-06_{dataset.lower()}_{split}_groups.json"

    if dataset == "Helpsteer2" and split == "train":
        filepath = f"data/dataset_detection/gpt-4o-2024-08-06_{dataset.lower()}_{split}_sampled_{groupSize}_grouped/gpt-4o-2024-08-06_{dataset.lower()}_{split}_sampled_groups.json"

    with open(filepath, 'r') as f:
            data = json.load(f)
    
    features = []
    labels = []

    for group in data:
        label = 1 if group['label'] == "LLM" else 0

        group_features = []

        if dataset == "ANTIQUE":
            group_features.extend(group['examples'][0]['ranking'])
        elif dataset == "Helpsteer3":
            group_features.append(group['examples'][0]['score'])
        elif dataset == "Helpsteer2":
            for dimension in datasets['Helpsteer2']:
                group_features.append(group['examples'][0][dimension])
        elif dataset == "Neurips":
            for dimension in datasets['Neurips']:
                group_features.append(group['examples'][0][dimension])
             
        features.append(group_features)
        labels.append(label)
    
    return np.array(features), np.array(labels)

def train_and_evaluate(X_train, y_train, X_test, y_test):
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(X_train, y_train)
    
    y_pred = classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return accuracy, f1


if __name__ == "__main__":
    accuracy_scores = []
    f1_scores = []

    for dataset in datasets.keys():
        X_train, y_train = load_data(dataset, "train", 1)
        X_test, y_test = load_data(dataset, "test", 1)

        accuracy, f1 = train_and_evaluate(X_train, y_train, X_test, y_test)

        accuracy_scores.append(accuracy)
        f1_scores.append(f1)

        print(f"Dataset: {dataset}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("-" * 30)
    
    print("Overall Performance:")
    print(f"Average Accuracy: {np.mean(accuracy_scores)}")
    print(f"Average F1 Score: {np.mean(f1_scores)}")