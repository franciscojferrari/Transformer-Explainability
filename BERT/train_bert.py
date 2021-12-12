from custom_bert import BertForSequenceClassification
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from transformers import PretrainedConfig, AutoTokenizer, get_scheduler, AdamW, TrainingArguments
from transformers.models.bert.configuration_bert import BertConfig
from datasets import load_dataset

if __name__ == "__main__":
    # Fix loading and preprocess dataset
    raw_imdb_dataset = load_dataset("imdb")
    model_config = BertConfig.from_json_file("BERT/BERT_params/movies_bert.json")
    tokenizer = AutoTokenizer.from_pretrained(model_config.bert_vocab)
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    tokenized_dataset = raw_imdb_dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(["text"])
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    tokenized_dataset.set_format("torch")

    train_dataset = tokenized_dataset["train"]
    eval_dataset = tokenized_dataset["test"]

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=1)#model_config.evidence_classifier["batch_size"])
    eval_dataloader = DataLoader(eval_dataset, batch_size=model_config.evidence_classifier["batch_size"])

    # Set up and train model
    model_config.num_labels = len(model_config.evidence_classifier["classes"])
    model = BertForSequenceClassification.from_pretrained(model_config.bert_dir)
    train_args = TrainingArguments(
        output_dir="BERT/hg_traind_dir",
        overwrite_output_dir=True,
        do_train=True,
        evaluation_strategy="no",
        learning_rate=model_config.evidence_classifier["lr"],
        num_train_epochs=model_config.evidence_classifier["epochs"],
        warmup_steps=model_config.evidence_classifier["warmup_steps"])


    early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(
        monitor="val_accuracy", 
        min_delta=0.00, 
        patience=model_config.evidence_classifier["patience"], 
        verbose=False, 
        mode="max")
    bert_system = BertSequenceClassificationSystem("bert-base-uncased", config=model_config)
    trainer = pl.Trainer(max_steps=1,
        # max_epochs = model_config.evidence_classifier["epochs"],
        gradient_clip_val=model_config.evidence_classifier["max_grad_norm"]) # Defaults to gradient_clip_algorithm="norm"
    trainer.fit(bert_system, train_dataloader, ckpt_path="BERT/BERT_checkpoints")
    try:
        trainer.save_checkpoint("BERT/BERT_checkpoints/final_checkpoint")
    except:
        pass
    bert_system.bert_classifier.save_pretrained("BERT_model")