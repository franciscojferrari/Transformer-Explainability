# Help notes by Sandor

## Relevance

The relevance for a block b, of layer n, is gotten by
![DeepTaylorDecomposition](sandor_note_imgs/DeepTaylorDecomposition.png)


## Attention

A^(b) = sofmax(Q^(b)*K^(b)/sqrt(d_h)) is the attention in each block of transformers b for query and key Q^(b) K^(b) in block b.

![transformer](sandor_note_imgs/transformer.png)

## Method
![method](sandor_note_imgs/method.png)

Operation: max(0, v) is denoted as v⁺

## Architechture relevant for classification fine tuning

![arch](sandor_note_imgs/BERT_NSP_TASK_Architechture_used_for_classification_fine_tuning.png)

## Relevance for layer zero (R⁽⁰⁾)

![R0](sandor_note_imgs/relevance_for_layer_zero.png)

