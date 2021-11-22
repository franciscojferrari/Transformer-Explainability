from custom_bert import BertForSequenceClassification
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from torchmetrics import Accuracy
from transformers import PretrainedConfig, AutoTokenizer, get_scheduler, AdamW
from transformers.models.bert.configuration_bert import BertConfig
from datasets import load_dataset, load_metric

class BertSequenceClassificationSystem(pl.LightningModule):
    def __init__(self, huggingface_model_name, config=None):
        super().__init__()
        if config is None:
            self.bert_classifier = BertForSequenceClassification.from_pretrained(huggingface_model_name)
        else:
            self.bert_classifier = BertForSequenceClassification.from_pretrained(huggingface_model_name, config=config)
        self.bert_classifier.train()
        self.acc = Accuracy()

    def forward(self, x):
        y = self.bert_classifier(**x)
        return torch.softmax(y.logits, dim=1)

    def training_step(self, batch, batch_idx):
        y = self.bert_classifier(**batch)
        self.log("train_loss", y.loss)
        self.log("train_accuracy", self.acc(torch.argmax(F.softmax(y.logits, dim=1), dim=1)), batch["labels"])
        return y.loss

    def validation_step(self, val_batch, val_batch_idx):
        y = self.bert_classifier(**val_batch)
        self.log("val_loss", y.loss)
        self.log("val_accuracy", self.acc(torch.argmax(F.softmax(y.logits, dim=1), dim=1)), val_batch["labels"])

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.bert_classifier.config.evidence_classifier["lr"])
        scheduler = get_scheduler(
            "constant", # Don't know if this is correct
            # It seems that Hila Chefer doesn't use a lr scheduler but has parameters for it in the config
            # Having constant lr is the same as having no 
            optimizer=optimizer,
            num_warmup_steps=self.bert_classifier.config.evidence_classifier["warmup_steps"]
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    
if __name__ == "__main__":
    # Fix loading and preprocess dataset
    train_dataset, val_dataset, eval_dataset = load_dataset("imdb", split=['train[:90%]', 'train[90%:]', 'test'])
    model_config = BertConfig.from_json_file("BERT/BERT_params/movies_bert.json")
    tokenizer = AutoTokenizer.from_pretrained(model_config.bert_vocab) # Change this if model changes
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    eval_dataset = eval_dataset.map(tokenize_function, batched=True)

    train_dataset = train_dataset.remove_columns(["text"])
    train_dataset = train_dataset.rename_column("label", "labels")
    train_dataset.set_format("torch")
    val_dataset = val_dataset.remove_columns(["text"])
    val_dataset = val_dataset.rename_column("label", "labels")
    val_dataset.set_format("torch")
    eval_dataset = eval_dataset.remove_columns(["text"])
    eval_dataset = eval_dataset.rename_column("label", "labels")
    eval_dataset.set_format("torch")

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=model_config.evidence_classifier["batch_size"])
    val_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=32) # Validation batch size is hardcoded =32 in Hila Chefers code for some reason
    eval_dataloader = DataLoader(eval_dataset, batch_size=model_config.evidence_classifier["batch_size"])

    # Set up and train model
    model_config.num_labels = len(model_config.evidence_classifier["classes"])
    early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(
        monitor="val_accuracy", 
        min_delta=0.00, 
        patience=model_config.evidence_classifier["patience"], 
        verbose=False, 
        mode="max")
    bert_system = BertSequenceClassificationSystem("bert-base-uncased", config=model_config)
    trainer = pl.Trainer(
        max_epochs = model_config.evidence_classifier["epochs"],
        default_root_dir="BERT/BERT_checkpoints",
        gradient_clip_val=model_config.evidence_classifier["max_grad_norm"]) # Defaults to gradient_clip_algorithm="norm"
    trainer.fit(bert_system, train_dataloader=train_dataloader, val_dataloaders=val_dataloader)#ckpt_path="BERT/BERT_checkpoints"
    try:
        trainer.save_checkpoint("BERT/BERT_checkpoints/final_checkpoint")
    except:
        pass
    bert_system.bert_classifier.save_pretrained("BERT_model")




