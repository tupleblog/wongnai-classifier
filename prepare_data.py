import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold


def save_json(ls, file_path):
    """
    Save list of dictionary to JSON
    """
    with open(file_path, 'w') as fp:
        fp.write('\n'.join(json.dumps(i) for i in ls))


def predict(comment):
    """
    Snippet to predict sentiment of Wongnai comment
    """
    from allennlp.models.archival import load_archive
    from allennlp.predictors.predictor import Predictor
    from wongnai.wongnai_reader import WongnaiDatasetReader
    from wongnai.wongnai_classifier import WongnaiCommentClassifier
    from wongnai.wongnai_predictor import WongnaiCommentPredictor
    from pythainlp import word_tokenize

    archive = load_archive('model.tar.gz')
    wongnai_predictor = Predictor.from_archive(archive, 'wongnai_predictor')
    prediction = wongnai_predictor.predict_json({"comment": word_tokenize(comment)})
    print(prediction)


def predict_ensemble(test_df):
    """
    Predict input ``test_df`` with columns ``review`` and ``rating``
    """
    instances = [wongnai_predictor._dataset_reader.text_to_instance(word_tokenize(review)) 
                 for review in list(test_df.review)]
    model_paths = glob('output_*/model.tar.gz')
    all_predicted_labels = []
    for model_path in model_paths:
        archive = load_archive(model_path) # load trained model
        wongnai_predictor = Predictor.from_archive(archive, 'wongnai_predictor')
        predicted_labels = [int(wongnai_predictor.predict_instance(instance)['predicted_label']) 
                                for instance in instances]
        all_predicted_labels.append(predicted_labels)
    all_predicted_labels = np.array(all_predicted_labels)
    predicted_labels_vote = mode(np.array(all_predicted_labels).T, axis=-1).mode.ravel()
    test_df['rating'] = predicted_labels_vote
    return test_df.drop('review', axis=1)



if __name__ == '__main__':
    """
    Read Wongnai dataset, you can download the zip file from https://github.com/wongnai/wongnai-corpus
    """
    wongnai_df = pd.read_csv('wongnai_dataset/w_review_train.csv', header=None).rename(columns={0: 'full_text'})
    wongnai_df['text'] = wongnai_df.full_text.map(lambda x: x.split(';')[0])
    wongnai_df['label'] = wongnai_df.full_text.map(lambda x: x.split(';')[-1])
    kfold = StratifiedKFold(n_splits=10)
    i = 0
    for train_index, test_index in kfold.split(wongnai_df.text, wongnai_df.label):
        train_df, valid_df = wongnai_df.iloc[train_index], wongnai_df.iloc[test_index]
        save_json([dict(r) for _, r in train_df.iterrows()], 'wongnai_dataset/training_{}.jsonl'.format(i))
        save_json([dict(r) for _, r in valid_df.iterrows()], 'wongnai_dataset/validation_{}.jsonl'.format(i))
        i += 1