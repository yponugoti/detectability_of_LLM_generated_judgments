import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

datasets = {
    "helpsteer2": ["helpfulness", "correctness", "coherence", "complexity", "verbosity"],
    "helpsteer3": ["score"],
    "neurips":    ["rating", "confidence", "soundness", "presentation", "contribution"],
    "ANTIQUE":    ["ranking"]
}

def load_data(dataset, split, groupSize):
    filepath = f"data/dataset_detection/gpt-4o-2024-08-06_{dataset.lower()}_{split}_{groupSize}_grouped/gpt-4o-2024-08-06_{dataset.lower()}_{split}_groups.json"

    if dataset == "helpsteer2" and split == "train":
        filepath = f"data/dataset_detection/gpt-4o-2024-08-06_{dataset.lower()}_{split}_sampled_{groupSize}_grouped/gpt-4o-2024-08-06_{dataset.lower()}_{split}_sampled_groups.json"

    with open(filepath, 'r') as f:
            data = json.load(f)
    
    rows = []

    for group in data:
        for example in group['examples']:
            row = {}
            row['label'] = 1 if example['label'] == "LLM" else 0
            row['group_id'] = example['group_id']

            if dataset == "ANTIQUE":
                row['query'] = str(example['query']).strip()
                row['response1'] = str(example['docs'][0]).strip()
                row['response2'] = str(example['docs'][1]).strip()
                row['response3'] = str(example['docs'][2]).strip()
                for i, rank in enumerate(example['ranking']):
                    row[f'ranking_{i+1}'] = rank
            elif dataset == "helpsteer3":
                row['response1'] = str(example['response1']).strip()
                row['response2'] = str(example['response2']).strip()
                row['score'] = example['score']
            elif dataset == "helpsteer2":
                row['prompt'] = str(example['prompt']).strip()
                row['response'] = str(example['response']).strip()
                for dimension in datasets['helpsteer2']:
                    row[f'feat_{dimension}'] = example[dimension]
            elif dataset == "neurips":
                content = example['content']
                content = "".join(content)
                row['content'] = str(content).strip()
                for dimension in datasets['neurips']:
                    row[f'feat_{dimension}'] = example[dimension]
        
            rows.append(row)
    
    return pd.DataFrame(rows)

def load_linguistic_features(dataset, split):
    filepath = f"data/features/linguistic_feature/{dataset}_{split}.csv"

    df = pd.read_csv(filepath)

    text_cols = ['query', 'response', 'response1', 'response2', 'prompt', 'content']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    cols_to_drop = [c for c in df.columns if "noun_verb_ratio" in c]
    if 'label' in df.columns:
        cols_to_drop.append('label')
    if 'ranking' in df.columns:
        cols_to_drop.append('ranking')
    df_clean = df.drop(columns=cols_to_drop)

    return df_clean

def load_llm_enhanced_features(dataset, split):
    filepath = f"data/features/llm_enhanced_features/{dataset}_{split}_Qwen3-8B.json"

    with open(filepath, 'r') as f:
        data = json.load(f)
    
    rows = []

    for item in data:
        row = {}

        if dataset == "ANTIQUE":
            row['query'] = str(item['query']).strip()
            row['response1'] = str(item['docs'][0]).strip()
            row['response2'] = str(item['docs'][1]).strip()
            row['response3'] = str(item['docs'][2]).strip()
            feat = item.get('llm_enhanced_feature')
            if feat:
                row['llm_r1_score'] = feat.get('Response1 Score', 0)
                row['llm_r2_score'] = feat.get('Response2 Score', 0)
                row['llm_r3_score'] = feat.get('Response3 Score', 0)
                    
                ranks = feat.get('Ranking', [])
                if isinstance(ranks, list):
                    for i, rank in enumerate(ranks):
                        row[f'llm_rank_{i}'] = rank
        elif dataset == "helpsteer3":
            row['response1'] = str(item.get('response1', '')).strip()
            row['response2'] = str(item.get('response2', '')).strip()

            r1 = item.get('llm_enhanced_feature_r1')
            r2 = item.get('llm_enhanced_feature_r2')
                
            if r1:
                for k, v in r1.items():
                    if "score" in k and isinstance(v, (int, float)): 
                        row[f'llm_r1_{k}'] = v
            if r2:
                for k, v in r2.items():
                    if "score" in k and isinstance(v, (int, float)): 
                        row[f'llm_r2_{k}'] = v
        elif dataset == "helpsteer2":
            row['prompt'] = str(item.get('prompt', '')).strip()
            row['response'] = str(item.get('response', '')).strip()
                
            feat = item.get('llm_enhanced_feature')
            if feat:
                for k, v in feat.items():
                    if "score" in k and isinstance(v, (int, float)): 
                        row[f'llm_{k}'] = v
        elif dataset == "neurips":
            content = item.get('content', '')
            if isinstance(content, list):
                content = "".join(content)
            row['content'] = str(content).strip()
                
            feat = item.get('llm_enhanced_feature')
            if feat:
                for k, v in feat.items():
                    if "score" in k and isinstance(v, (int, float)): 
                        row[f'llm_{k}'] = v
        
        rows.append(row)
    
    return pd.DataFrame(rows)

def train_and_evaluate(X_train, y_train, X_test, y_test):
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(X_train, y_train)
    
    y_pred = classifier.predict_proba(X_test)[:, 1]

    df = y_test[['group_id', 'label']].copy()
    df['prob_llm'] = y_pred

    group_aggregation = df.groupby('group_id').agg({
        "label": 'first',
        "prob_llm": 'mean'
    }).reset_index()

    group_aggregation['final_pred'] = (group_aggregation['prob_llm'] >= 0.5).astype(int)

    accuracy = accuracy_score(group_aggregation['label'], group_aggregation['final_pred'])
    f1 = f1_score(group_aggregation['label'], group_aggregation['final_pred'])

    return accuracy, f1


if __name__ == "__main__":
    accuracy_scores = []
    f1_scores = []

    for dataset in datasets.keys():
        X_train_ling = load_linguistic_features(dataset, "train")
        X_test_ling = load_linguistic_features(dataset, "test")

        df_llm_train = load_llm_enhanced_features(dataset, "train")
        df_llm_test = load_llm_enhanced_features(dataset, "test")

        if dataset == "ANTIQUE":
            merge_on = ['query', 'response1', 'response2', 'response3']
        elif dataset == "helpsteer3":
            merge_on = ['response1', 'response2']
        elif dataset == "helpsteer2":
            merge_on = ['prompt', 'response']
        elif dataset == "neurips":
            merge_on = ['content']

        for groupSize in [1, 2, 4, 8, 16]:
            X_train_base = load_data(dataset, "train", groupSize)
            X_test_base = load_data(dataset, "test", groupSize)

            train_merged = pd.merge(X_train_base, X_train_ling, on=merge_on, how='inner')
            train_merged = pd.merge(train_merged, df_llm_train, on=merge_on, how='inner')
            test_merged = pd.merge(X_test_base, X_test_ling, on=merge_on, how='inner')
            test_merged = pd.merge(test_merged, df_llm_test, on=merge_on, how='inner')

            y_train = train_merged['label'].values
            y_test = test_merged['label'].values

            metadata_cols = ["query", "response", "response1", "response2", "response3", 
                            "prompt", "content", "context", "ranking", "label", "group_id"]

            X_train_aug = train_merged.drop(columns=[c for c in train_merged.columns if c in metadata_cols], errors='ignore').select_dtypes(include=[np.number]).values
            X_test_aug = test_merged.drop(columns=[c for c in test_merged.columns if c in metadata_cols], errors='ignore').select_dtypes(include=[np.number]).values

            accuracy, f1 = train_and_evaluate(X_train_aug, y_train, X_test_aug, test_merged)

            accuracy_scores.append(accuracy)
            f1_scores.append(f1)

            print(f"Dataset: {dataset} of size {groupSize}")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print("-" * 30)
    
    print("Overall Performance:")
    print(f"Average Accuracy: {np.mean(accuracy_scores)}")
    print(f"Average F1 Score: {np.mean(f1_scores)}")