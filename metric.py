from seqeval.metrics import f1_score, accuracy_score
from typing import Dict, List

tags = ['B-LOC', 'B-MISC', 'B-ORG', 'B-PER', 'I-LOC', 'I-MISC', 'I-ORG', 'I-PER', 'O']

tag2id = {tag: i for i, tag in enumerate(tags)}
id2tag = {i: tag for tag, i in tag2id.items()}

def calc_f1(predictions: List[List[int]], labels: List[List[int]]):
    """
    :params:
    predictions: list of lists of predicted labels
    labels: list of lists of ground truth labels
    """
    text_labels = [[id2tag[l] for l in label if l != -100] for label in labels]
    text_predictions = []
    for i in range(len(text_labels)):
        # +1 because we skip the first ([CLS]) token
        sample_text_preds = [id2tag[predictions[i][j + 1]] for j in range(len(text_labels[i]))]
        text_predictions.append(sample_text_preds)

    return f1_score(text_labels, text_predictions)