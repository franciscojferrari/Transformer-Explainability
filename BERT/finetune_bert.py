from custom_bert import BertForSequenceClassification
import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl

class BertSequenceClassificationSystem(pl.LightningModule):
    def __init__(self, huggingface_model_name, num_labels=2):
        super().__init__()
        self.bert_classifier = BertForSequenceClassification.from_pretrained(huggingface_model_name, num_labels=num_labels)
        self.bert_classifier.train()

    def forward(self, x):
        y = self.bert_classifier(**x)
        return torch.softmax(y.logits, dim=1)

    def training_step(self, batch, batch_idx):
        y = self.bert_classifier(**batch)
        self.log("train_loss", y.loss)
        return y.loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    
if __name__ == "__main__":
    from datasets import load_dataset
    raw_imdb_dataset = load_dataset("imdb")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased") # Change this if model changes

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    tokenized_dataset = raw_imdb_dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(["text"])
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    tokenized_dataset.set_format("torch")

    train_dataset = tokenized_dataset["train"]
    eval_dataset = tokenized_dataset["test"]

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=1)
    eval_dataloader = DataLoader(eval_dataset, batch_size=1)

    bert_system = BertSequenceClassificationSystem("bert-base-uncased")
    trainer = pl.Trainer()
    trainer.fit(bert_system, train_dataloader)




