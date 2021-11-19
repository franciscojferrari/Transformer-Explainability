import torch
from typing import Optional
from torch import autograd
from torch.autograd import grad
from torch.nn import functional as F
from torch.nn import modules

def backward_hook(module, grad_in, grad_out):
    module.grad_wrt_input = grad_in
    module.grad_wrt_output = grad_out

# Input is only non-positional arguments to forward function
def forward_hook(module, input, output):
    if type(input[0]) in (list, tuple):
        module.module_input = []
        for i in input[0]:
            x = i.detach()
            x.requires_grad = True
            module.module_input.append(x)
    else:
        module.module_input = input[0].detach().float()
        module.module_input.requires_grad = True

class Rel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.grad_wrt_input = None
        self.grad_wrt_output = None
        self.module_output = None
        self.module_input = None
        self.eval_and_hook()

    def eval_and_hook(self):
        self.eval()
        self.register_full_backward_hook(backward_hook)
        self.register_forward_hook(forward_hook)

    def relevance_propagation(self, prev_rel):
        return prev_rel

class RelDeepTaylor(Rel):
    def relevance_propagation(self, prev_rel):
        output = self.forward(self.module_input)
        frac = prev_rel / output
        grad = torch.autograd.grad(output, self.module_input, frac, retain_graph=True)[0]
        
        if torch.is_tensor(self.module_input):
            return self.module_input * grad
        else:
            rels = []
            for input in self.module_input:
                rels.append(input * grad)
            return rels

class RelAlphaBeta(Rel):
    def alpha_beta(self, module_input, prev_rel, retain_graph, alpha, beta):
        pos_weights = torch.where(self.weight > 0, self.weight, torch.zeros(1, dtype=self.weight.dtype))
        neg_weights = torch.where(self.weight < 0, self.weight, torch.zeros(1, dtype=self.weight.dtype))
        pos_inputs = torch.where(module_input > 0, module_input, torch.zeros(1, dtype=module_input.dtype))
        neg_inputs = torch.where(module_input < 0, module_input, torch.zeros(1, dtype=module_input.dtype))

        pos_pos = F.linear(pos_inputs, pos_weights).clamp(min=1e-9)
        pos_neg = F.linear(pos_inputs, neg_weights).clamp(min=1e-9)
        neg_pos = F.linear(neg_inputs, pos_weights).clamp(min=1e-9)
        neg_neg = F.linear(neg_inputs, neg_weights).clamp(min=1e-9)

        pos_rel = \
            pos_inputs * torch.autograd.grad(pos_pos, pos_inputs, prev_rel / pos_pos, retain_graph=retain_graph)[0] \
            + neg_inputs * torch.autograd.grad(neg_neg, neg_inputs, prev_rel / neg_neg)[0]
        neg_rel = \
            pos_inputs * torch.autograd.grad(pos_neg, pos_inputs, prev_rel / pos_neg)[0] \
            + neg_inputs * torch.autograd.grad(neg_pos, neg_inputs, prev_rel / neg_pos)[0]

        return alpha * pos_rel - beta * neg_rel

    def relevance_propagation(self, prev_rel, retain_graph, alpha=0.5, beta=0.5,):
        if torch.is_tensor(self.module_input):
            return self.alpha_beta(self.module_input, prev_rel, retain_graph, alpha, beta)
        else:
            rels = []
            for module_input in self.module_input:
                rels.append(self.alpha_beta(module_input, prev_rel, retain_graph, alpha, beta))
            return rels

class RelGeLu(RelAlphaBeta):
    def alpha_beta(self, module_input, prev_rel, retain_graph, alpha, beta):
        pos_weights = torch.where(self.weight > 0, self.weight, torch.zeros(1, dtype=self.weight.dtype))
        neg_weights = torch.where(self.weight < 0, self.weight, torch.zeros(1, dtype=self.weight.dtype))
        pos_inputs = torch.where(module_input > 0, module_input, torch.zeros(1, dtype=module_input.dtype))
        neg_inputs = torch.where(module_input < 0, module_input, torch.zeros(1, dtype=module_input.dtype))

        pos_pos = F.linear(pos_inputs, pos_weights)
        pos_neg = F.linear(pos_inputs, neg_weights)
        neg_pos = F.linear(neg_inputs, pos_weights)
        neg_neg = F.linear(neg_inputs, neg_weights)

        pos_pos = torch.where(pos_pos > 0, pos_pos, torch.zeros(1, dtype=self.weight.dtype)).clamp(min=1e-9)
        pos_neg = torch.where(pos_neg > 0, pos_neg, torch.zeros(1, dtype=self.weight.dtype)).clamp(min=1e-9)
        neg_pos = torch.where(neg_pos > 0, neg_pos, torch.zeros(1, dtype=self.weight.dtype)).clamp(min=1e-9)
        neg_neg = torch.where(neg_neg > 0, neg_neg, torch.zeros(1, dtype=self.weight.dtype)).clamp(min=1e-9)


        pos_rel = \
            pos_inputs * torch.autograd.grad(pos_pos, pos_inputs, prev_rel / pos_pos, retain_graph=retain_graph)[0] \
            + neg_inputs * torch.autograd.grad(neg_neg, neg_inputs, prev_rel / neg_neg)[0]
        neg_rel = \
            pos_inputs * torch.autograd.grad(pos_neg, pos_inputs, prev_rel / pos_neg)[0] \
            + neg_inputs * torch.autograd.grad(neg_pos, neg_inputs, prev_rel / neg_pos)[0]

        return alpha * pos_rel - beta * neg_rel

class Embedding(torch.nn.Embedding, Rel):
    pass

class LayerNorm(torch.nn.LayerNorm, Rel):
    pass

class Dropout(torch.nn.Dropout, Rel):
    pass

class Linear(torch.nn.Linear, RelAlphaBeta):
    pass

class Softmax(torch.nn.Softmax, Rel):
    pass

class einsum(Rel):
    def forward(self, equation, *operands):
        return torch.einsum(equation, *operands)

if __name__ == "__main__":
    module = Linear(16, 4, bias=True)
    input = torch.tensor([[1.0,2.0,4.0,5.0,1.0,2.0,4.0,5.0,1.0,2.0,4.0,5.0,1.0,2.0,4.0,5.0],[4,3,2,9,4,3,2,9,4,3,2,9,4,3,2,9]], requires_grad=True)
    output = module(input)
    module2 = Linear(4, 1)
    loss = torch.nn.functional.mse_loss(torch.nn.functional.softmax(module2(output)), torch.tensor([[1.0], [0]])).backward(retain_graph=True)
    rel = module.relevance_propagation(torch.tensor([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]), retain_graph=True)

    print("Done!")