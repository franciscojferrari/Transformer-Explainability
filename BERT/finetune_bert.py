from custom_bert import BertForSequenceClassification
from BertClassExplanation import BertForSequenceClassificationExplanator
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from torchmetrics import Accuracy, F1
from transformers import PretrainedConfig, AutoTokenizer, get_scheduler, AdamW
from transformers.models.bert.configuration_bert import BertConfig
from datasets import load_dataset, load_metric
import argparse
import os

class BertSequenceClassificationSystem(pl.LightningModule):
    def __init__(self, huggingface_model_name, config=None):
        super().__init__()
        if config is None:
            self.bert_classifier = BertForSequenceClassification.from_pretrained(huggingface_model_name)
        else:
            self.bert_classifier = BertForSequenceClassification.from_pretrained(huggingface_model_name, config=config)
        self.bert_classifier.train()
        self.acc = Accuracy()
        self.f1 = F1()
        self.explanator = BertForSequenceClassificationExplanator(self.bert_classifier)


    def forward(self, x):
        y = self.bert_classifier(**x)
        return torch.softmax(y.logits, dim=1)

    def training_step(self, batch, batch_idx):
        y = self.bert_classifier(**batch)
        logits = y.logits
        self.log("train_loss", y.loss)
        if y.logits.dim()-1 != batch["labels"].dim():
            logits = logits.squeeze(1)
            self.log("train_f1", self.f1(y.logits,  batch["labels"]))
            self.log("train_accuracy", self.acc(torch.argmax(F.softmax(y.logits, dim=1), dim=1, keepdim=True)), batch["labels"])
        return y.loss

    def validation_step(self, val_batch, val_batch_idx):
        y = self.bert_classifier(**val_batch)
        logits = y.logits
        self.log("val_loss", y.loss)
        if y.logits.dim()-1 != val_batch["labels"].dim():
            logits = logits.squeeze(1)
        self.log("train_f1", self.f1(logits,  val_batch["labels"]))
        self.log("val_accuracy", self.acc(torch.argmax(F.softmax(logits, dim=1), dim=1), val_batch["labels"]))

    def test_step(self, test_batch, test_batch_id):
        torch.set_grad_enabled(True)
        hard_rationale = test_batch.pop("hard_rationale")
        explanation, one_hot_pred = self.explanator.generate_explanation(**test_batch)
        return explanation, one_hot_pred, hard_rationale
    
    # def test_step_end(self, test_step_outputs):
    #     explanations = [test_step_outputs[0] for _ in range(len(test_step_outputs))]
    #     one_hot_pred = [test_step_outputs[1] for _ in range(len(test_step_outputs))]
    #     hard_rationale = [test_step_outputs[2] for _ in range(len(test_step_outputs))]
        
    #     print("done")

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.bert_classifier.config.evidence_classifier["lr"])
        scheduler = get_scheduler(
            "constant", # Don't know if this is correct
            # It seems that Hila Chefer doesn't use a lr scheduler but has parameters for it in the config
            # Having constant lr is the same as having no scheduler
            optimizer=optimizer,
            num_warmup_steps=self.bert_classifier.config.evidence_classifier["warmup_steps"]
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Custom BERT model for pytorch.')
    parser.add_argument("-p", "--bert-params", type=str,
                    help="Path to bert params (json file).")
    parser.add_argument("-c", "--pytorch-lightning-checkpoint-dir", type=str,
                    help="Path to the pytorch checkpoint directory (Can be a brand new or earliear directory).")
    parser.add_argument("-ckpt", "--pytorch-lightning-checkpoint-path", type=str,
                    help="Path to a specific pytorch lightning checkpoint.")
    parser.add_argument("-t", "--train", action='store_true', help="Flag to train the model. Otherwise you test the model.")
    args = parser.parse_args()
    if args.pytorch_lightning_checkpoint_dir is None:
        parser.error('You have to specify checkpoint directory for pytorch lightning')
    if args.bert_params is None:
        parser.error("You have to specify the bert_params")
    if not os.path.isdir(args.pytorch_lightning_checkpoint_dir):
        parser.error("{} is not a folder or does not exist".format(args.pytorch_lightning_checkpoint_dir))

    if args.train:
        print("Starting to train...")
        try:
            os.mkdir("BertForSequenceClassification")
        except FileExistsError:
            pass
        # Fix loading and preprocess dataset
        train_dataset, val_dataset = load_dataset("movie_rationales", split=['train', 'validation'])
        model_config = BertConfig.from_json_file(args.bert_params)
        tokenizer = AutoTokenizer.from_pretrained(model_config.bert_vocab)
        def tokenize_function(examples):
            return tokenizer(examples["review"], padding="max_length", truncation=True)
        def label_parse(examples):
            return {"label": [1 if examples["label"][i] == 1 else 0 for i in range(len(examples["label"]))]}
        train_dataset = train_dataset.map(tokenize_function, batched=True)
        train_dataset = train_dataset.map(label_parse, batched=True)
        val_dataset = val_dataset.map(tokenize_function, batched=True)
        val_dataset = val_dataset.map(label_parse, batched=True)

        # train_dataset.reomve_columns("review")
        train_dataset = train_dataset.remove_columns(["evidences", "review"])
        train_dataset = train_dataset.rename_column("label", "labels")
        train_dataset.set_format("torch")
        val_dataset = val_dataset.remove_columns(["evidences", "review"])
        val_dataset = val_dataset.rename_column("label", "labels")
        val_dataset.set_format("torch")

        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=model_config.evidence_classifier["batch_size"])
        val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=model_config.evidence_classifier["batch_size"]) # Validation batch size is hardcoded =32 in Hila Chefers code for some reason

        # Set up and train model
        model_config.num_labels = len(model_config.evidence_classifier["classes"])
        early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(
            monitor="val_accuracy", 
            min_delta=0.00, 
            patience=model_config.evidence_classifier["patience"], 
            verbose=False, 
            mode="max")
        bert_system = BertSequenceClassificationSystem(model_config.bert_dir, config=model_config)
        trainer = pl.Trainer(
            max_epochs = model_config.evidence_classifier["epochs"],
            default_root_dir=args.pytorch_lightning_checkpoint_dir,
            gradient_clip_val=model_config.evidence_classifier["max_grad_norm"]) # Defaults to gradient_clip_algorithm="norm"
        trainer.fit(bert_system, train_dataloader=train_dataloader, val_dataloaders=val_dataloader, ckpt_path=args.pytorch_lightning_checkpoint_path)
        try:
            trainer.save_checkpoint(os.path.join(args.pytorch_lightning_checkpoint_dir, "final_checkpoint"))
        except:
            pass
        bert_system.bert_classifier.save_pretrained("BertForSequenceClassification")
    else:
        print("Starting to test...")
        # if args.pytorch_lightning_checkpoint_path is None:
        #     parser.error("If not training, you have to specify a checkpoint to test the model from.")
        train_dataset = load_dataset("movie_rationales", split="train[:5]")
        model_config = BertConfig.from_json_file(args.bert_params)
        tokenizer = AutoTokenizer.from_pretrained(model_config.bert_vocab)
        def get_hard_rationale(data):
            hard_rationales = []
            for i in range(len(data["review"])):
                hard_rationale = []
                for j in range(len(data["evidences"][i])):
                    token_split = data["evidences"][i][j].split()
                    for k in range(len(token_split)):
                        index_in_review = data["review"][i].find(token_split[k])
                        if index_in_review not in hard_rationale:
                            hard_rationale.append(index_in_review)
                hard_rationales.append(hard_rationale)
            return {"hard_rationale": hard_rationales}
        def tokenize_function(data):
            return tokenizer(data["review"], padding="max_length", truncation=True)
        def label_parse(examples):
            return {"label": [1 if examples["label"][i] == 1 else 0 for i in range(len(examples["label"]))]}
        train_dataset = train_dataset.map(get_hard_rationale, batched=True)
        train_dataset = train_dataset.map(tokenize_function, batched=True)
        train_dataset = train_dataset.map(label_parse, batched=True)
        train_dataset = train_dataset.remove_columns(["evidences", "review"])
        train_dataset = train_dataset.rename_column("label", "labels")
        train_dataset.set_format("torch")
        def collate(batch):
            collated_batch = batch[0]
            for k in collated_batch.keys():
                if k == "hard_rationale":
                    collated_batch[k] = [collated_batch[k]]
                elif k == "labels":
                    collated_batch[k] = [collated_batch[k].item()]
                else:
                    collated_batch[k] = collated_batch[k].unsqueeze(0)
            for i in range(1, len(batch)):
                collated_batch["attention_mask"] = torch.vstack((collated_batch["attention_mask"], batch[i]["attention_mask"].unsqueeze(0)))
                collated_batch["input_ids"] = torch.vstack((collated_batch["input_ids"], batch[i]["input_ids"].unsqueeze(0)))
                collated_batch["labels"].append(batch[i]["labels"].item())
                collated_batch["token_type_ids"] = torch.vstack((collated_batch["token_type_ids"], batch[i]["token_type_ids"].unsqueeze(0)))
                collated_batch["hard_rationale"].append(batch[i]["hard_rationale"])
            for k in collated_batch.keys():
                if k == "labels":
                    collated_batch[k] = torch.tensor(collated_batch[k])
                elif not k == "hard_rationale":
                    collated_batch[k] = collated_batch[k].float().requires_grad_(True)
            return collated_batch
        train_dataloader = DataLoader(train_dataset, batch_size=model_config.evidence_classifier["batch_size"], collate_fn=collate)

        bert_system = BertSequenceClassificationSystem(model_config.bert_dir, config=model_config)
        trainer = pl.Trainer(
            max_epochs = model_config.evidence_classifier["epochs"],
            default_root_dir=args.pytorch_lightning_checkpoint_dir,
            gradient_clip_val=model_config.evidence_classifier["max_grad_norm"])
        trainer.test(bert_system, train_dataloader, ckpt_path=args.pytorch_lightning_checkpoint_path)



        # Not finished!





