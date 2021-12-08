import os
from datasets import load_dataset
import numpy as np

if __name__ == "__main__":
    files = os.listdir("BERT/BERT_preds")
    preds = np.empty(len(files))
    for i, file in enumerate(files):
        file_path = os.path.join("BERT/BERT_preds", file)
        preds[i] = np.loadtxt(file_path)
    dataset = load_dataset("movie_rationales", split="test")
    truth = np.array(dataset["label"])
    print("Acc: {}".format((preds.astype("int") == truth.astype("int")).astype("int").sum() / truth.shape[0]))

