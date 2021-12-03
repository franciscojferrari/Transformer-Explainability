import datasets
import torch
from datasets import load_dataset
from torchmetrics import F1
from typing import Any, Callable, Dict, List, Set, Tuple
from collections import Counter, defaultdict, namedtuple
from dataclasses import dataclass
from eraser_benchmark.utils import load_datasets, load_documents, Annotation
from eraser_benchmark.metrics import Rationale, score_hard_rationale_predictions
import os
import numpy as np
from itertools import chain

# What Hila Chefer does is that she takes _, indicies = expl.topk on the explnations,
# then she sets start_token=indicies[i], end_token=indicies[i]+1

def proccess_predictions(explanations_folder, preds_folder, k_list=range(80), class_names=["NEG", "POS"]):
    try:
        os.mkdir("BERT/BERT_annotations/")
        # for k in k_list:
        #     os.mkdir
    except FileExistsError:
        pass
    rationales_list = [[] for _ in k_list]
    file_names = os.listdir("BERT_explanations")
    file_names.sort(key=lambda x: int(x[:-4]))
    for i, file_name in enumerate(file_names):
        expl = np.loadtxt(os.path.join(explanations_folder, file_name), delimiter=" ")
        # one_hot_preds = int(np.loadtxt(os.path.join(preds_folder, file_name), delimiter=" ").item())
        for k in k_list:
            if i < 800:
                id = "negR_{0:03}.txt".format(i)
            else:
                id = "posR_{0:03}.txt".format(i-800)
            rationale_tokens = (-expl).argsort()[:k+1]
            for rationale_token in rationale_tokens.tolist():
                rationales_list[k].append(Rationale(id, id, rationale_token, rationale_token+1))
    return rationales_list

if __name__ == "__main__":
    train, _, _ = load_datasets("BERT/movies_dataset",)
    truth = list(chain.from_iterable(Rationale.from_annotation(ann) for ann in train))
    pred = proccess_predictions("BERT_explanations", "BERT_preds")
    scores = []
    for k in range(80):
        scores.append(score_hard_rationale_predictions(truth, pred[k]))
    pass