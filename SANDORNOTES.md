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

