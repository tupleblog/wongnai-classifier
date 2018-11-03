import json
import pandas as pd
from sklearn.model_selection import train_test_split

def save_json(ls, file_path):
    """
    Save list of dictionary to JSON
    """
    with open(file_path, 'w') as fp:
        fp.write('\n'.join(json.dumps(i) for i in ls))


if __name__ == '__main__':
    """
    Read Wongnai dataset with labels 0, 1 from `sent_raw.csv` file and save to JSON format
    """
    sent_df = pd.read_csv('sent_raw.csv')
    train_df, valid_df = train_test_split(sent_df, test_size=0.25, stratify=sent_df.target)
    save_json([dict(r) for _, r in train_df.iterrows()], 'wongnai_dataset/training.jsonl')
    save_json([dict(r) for _, r in valid_df.iterrows()], 'wongnai_dataset/validation.jsonl')