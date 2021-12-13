# Old commands (outdated)
```python BERT/finetune_bert.py -g 1 -p BERT/BERT_params/movies_bert.json -t -n 4```

```python -O BERT/finetune_bert.py -g 1 -p BERT/BERT_params/movies_bert.json -n 4 -b BertForSequenceClassification```

# Inference command for Hila Chefer model

Put their model ckpt in ```classier```. It is downloaded from https://drive.google.com/file/d/1kGMTr69UWWe70i-o2_JfjmWDQjT66xwQ/view?usp=sharing.

```python BERT/finetune_bert.py -p BERT/BERT_params/movies_bert.json -g 1 -ckpt classifier/classifier.pt -n 2```

# Quick train accuracy

```python BERT/acc.py```

# Token F1 experiment

Download their dataset from https://drive.google.com/file/d/11faFLGkc0hkw3wrGTYJBr1nIvkRb189F/view?usp=sharing and unzip in top level of git folder.

```python BERT/bert_utils.py```