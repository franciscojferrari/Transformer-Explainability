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
import json
import argparse

# What Hila Chefer does is that she takes _, indicies = expl.topk on the explnations,
# then she sets start_token=indicies[i], end_token=indicies[i]+1

def proccess_predictions(explanations_folder, docids, k_list=range(10, 80+10, 10), class_names=["NEG", "POS"]):
    rationales_list = [[] for _ in range(len(k_list))]
    file_names = os.listdir(explanations_folder)
    file_names.sort(key=lambda x: int(x[:-4]))
    docids = list(docids)
    doclen = len(docids)
    docids.sort(key=lambda x: int(x[5:8]) if "neg" in x else int(x[5:8]) + doclen)
    for i, file_name in enumerate(file_names):
        expl = np.loadtxt(os.path.join(explanations_folder, file_name), delimiter=" ")
        for j, k in enumerate(k_list):
            id = docids[i]
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

def json_format_dict(explanations_folder, docids, k_list=range(10, 80+10, 10), class_names=["NEG", "POS"]):
    rationales_list = [[] for _ in range(len(k_list))]
    file_names = os.listdir(explanations_folder)
    file_names.sort(key=lambda x: int(x[:-4]))
    docids = list(docids)
    doclen = len(docids)
    docids.sort(key=lambda x: int(x[5:8]) if "neg" in x else int(x[5:8]) + doclen)
    for i, file_name in enumerate(file_names):
        expl = np.loadtxt(os.path.join(explanations_folder, file_name), delimiter=" ")
        for j, k in enumerate(k_list):
            id = docids[i]
            rationale_tokens = (-expl).argsort()[:k]
            hard_rationale_predictions = []
            for rationale_token in rationale_tokens.tolist():
                hard_rationale_predictions.append({"start_token": rationale_token, "end_token": rationale_token + 1})
                # rationales_list[j].append(Rationale(id, id, rationale_token, rationale_token+1))
            rationales = [{"docid": id, "hard_rationale_predictions": hard_rationale_predictions}]
            jsonl_entry = {"annotation_id": id, "rationales": rationales}
            rationales_list[j].append(jsonl_entry)
    return rationales_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Do token-F1 experiment from ERASER.')
    parser.add_argument("--split", required=True, type=str, help="Split to check metrics by.")
    parser.add_argument("--data-dir", required=True, type=str, help="Directory of dataset.")
    parser.add_argument("--bert-explanations-dir", required=True, type=str, help="Directory of explanations")
    args = parser.parse_args()
    train, val, test = load_datasets(args.data_dir)
    if args.split == "train":
        dataset = train
    elif args.split == "test":
        dataset = test
    elif args.split == "val":
        dataset = val
    else:
        parser.error("Unrecognized split of data.")
    truth = list(chain.from_iterable(Rationale.from_annotation(ann) for ann in dataset))
    docids = set([rat.docid for rat in truth])
    truth = make_rationales_hard(truth)
    k_list = list(range(10, 80+10, 10))
    pred = proccess_predictions(args.bert_explanations_dir, docids, k_list=k_list)
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
    pred = json_format_dict(args.bert_explanations_dir, docids)
    try:
        os.mkdir("BERT/BERT_json_res")
    except FileExistsError:
        pass
    for i in range(len(pred)):
        with open("BERT/BERT_json_res/{}.jsonl".format(k_list[i]), "w") as fh:
            json.dump(pred[i], fh)


"""
{
        "annotation_id": str, required
        these classifications *must not* overlap
        "rationales": List[
            {
                "docid": str, required
                "hard_rationale_predictions": List[{
                    "start_token": int, inclusive, required
                    "end_token": int, exclusive, required
                }], optional,
                token level classifications, a value must be provided per-token
                in an ideal world, these correspond to the hard-decoding above.
            }
        ],
        the classification the model made for the overall classification task
        "classification": str, optional
        A probability distribution output by the model. We require this to be normalized.
        "classification_scores": Dict[str, float], optional
        The next two fields are measures for how faithful your model is (the
        rationales it predicts are in some sense causal of the prediction), and
        how sufficient they are. We approximate a measure for comprehensiveness by
        asking that you remove the top k%% of tokens from your documents,
        running your models again, and reporting the score distribution in the
        "comprehensiveness_classification_scores" field.
        We approximate a measure of sufficiency by asking exactly the converse
        - that you provide model distributions on the removed k%% tokens.
        'k' is determined by human rationales, and is documented in our paper.
        You should determine which of these tokens to remove based on some kind
        of information about your model: gradient based, attention based, other
        interpretability measures, etc.
        scores per class having removed k%% of the data, where k is determined by human comprehensive rationales
        "comprehensiveness_classification_scores": Dict[str, float], optional
        scores per class having access to only k%% of the data, where k is determined by human comprehensive rationales
        "sufficiency_classification_scores": Dict[str, float], optional
        the number of tokens required to flip the prediction - see "Is Attention Interpretable" by Serrano and Smith.
        "tokens_to_flip": int, optional
        "thresholded_scores": List[{
            "threshold": float, required,
            "comprehensiveness_classification_scores": like "classification_scores"
            "sufficiency_classification_scores": like "classification_scores"
        }], optional. if present, then "classification" and "classification_scores" must be present
    }
"""