import torch
from typing import Optional

class Embedding(torch.nn.Embedding):
    pass

class LayerNorm(torch.nn.LayerNorm):
    pass

class Dropout(torch.nn.Dropout):
    pass

class Linear(torch.nn.Linear):
    pass

class einsum:
    def forward(self, equation, *operands):
        return torch.einsum(equation, *operands)

if __name__ == "__main__":
    emb = Embedding(10, 3, padding_idx=0)
    input = torch.LongTensor([[1,2,4,5],[4,3,2,9]])
    output = emb(input)
    print("Done!")