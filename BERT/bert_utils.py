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

def proccess_predictions(explanations_folder, k_list=range(10, 80+10, 10), class_names=["NEG", "POS"]):
    try:
        os.mkdir("BERT/BERT_annotations/")
        # for k in k_list:
        #     os.mkdir
    except FileExistsError:
        pass
    rationales_list = [[] for _ in range(len(k_list))]
    file_names = os.listdir("BERT_explanations")
    file_names.sort(key=lambda x: int(x[:-4]))
    for i, file_name in enumerate(file_names):
        expl = np.loadtxt(os.path.join(explanations_folder, file_name), delimiter=" ")
        for j, k in enumerate(k_list):
            if i < 800:
                id = "negR_{0:03}.txt".format(i)
            else:
                id = "posR_{0:03}.txt".format(i-800)
            rationale_tokens = (-expl).argsort()[:k]
            for rationale_token in rationale_tokens.tolist():
                rationales_list[j].append(Rationale(id, id, rationale_token, rationale_token+1))
    return rationales_list

def make_rationales_hard(soft_rationale_list):
    hard_rationale_list = []
    for soft_rationale in soft_rationale_list:
        for token in range(soft_rationale.start_token, soft_rationale.end_token):
            hard_rationale_list.append(Rationale(soft_rationale.ann_id, soft_rationale.docid, token, token+1))
    return hard_rationale_list

if __name__ == "__main__":
    train, _, _ = load_datasets("movies_dataset",)
    truth = list(chain.from_iterable(Rationale.from_annotation(ann) for ann in train))
    truth = make_rationales_hard(truth)
    k_list = list(range(10, 80+10, 10))
    pred = proccess_predictions("BERT_explanations", k_list=k_list)
    f1_micro_scores = np.zeros((2, len(pred)))
    f1_macro_scores = np.zeros((2, len(pred)))
    f1_micro_scores[0, :] = np.array(k_list)[:]
    f1_macro_scores[0, :] = np.array(k_list)[:]


    for i in range(len(pred)):
        score = score_hard_rationale_predictions(truth, pred[i])
        f1_micro_scores[1, i] = score['instance_micro']['f1']
        f1_macro_scores[1, i] = score['instance_macro']['f1']
        print("K={}: {}".format(k_list[i], score))
    np.savetxt("f1_micro_scores.csv", f1_micro_scores, delimiter=",")
    np.savetxt("f1_macro_scores.csv", f1_macro_scores, delimiter=",")