import os
from datasets import load_dataset
import numpy as np

if __name__ == "__main__":
    files = os.listdir("BERT_preds")
    preds = np.empty(len(files))
    for i, file in enumerate(files):
        file_path = os.path.join("BERT_preds", file)
        preds[i] = np.loadtxt(file_path)
    train_dataset = load_dataset("movie_rationales", split="train")
    truth = np.array(train_dataset["label"])
    preds = np.where(preds == -1, 0, 1)
    print("Acc: {}".format((preds == truth).sum() / truth.shape[0]))

