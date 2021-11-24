import torch
from typing import Optional
from torch import autograd
from torch.autograd import grad
from torch.nn import functional as F
from torch.nn import modules

def divide_add_epsilon(num, den, epsilon=1e-9):
    # Add epsilon to denominator
    den_eps = den + epsilon
    # Where denominator == -epsilon, add 2*epsilon instead
    return num / torch.where(den_eps == 0, torch.tensor(epsilon), den_eps)

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
        self.hook()

    def hook(self):
        self.register_full_backward_hook(backward_hook)
        self.register_forward_hook(forward_hook)

    def relevance_propagation(self, prev_rel):
        return prev_rel

class RelDeepTaylor(Rel):
    def relevance_propagation(self, prev_rel, **kwargs):
        output = self.forward(self.module_input)
        frac = prev_rel / output
        grad = torch.autograd.grad(output, self.module_input, frac, retain_graph=True)
        
        if torch.is_tensor(self.module_input):
            return self.module_input * grad
        else:
            rels = []
            for i, input in enumerate(self.module_input):
                rels.append(input * grad[i])
            return rels

class RelEpsilon(Rel):
    def relevance_propagation(self, prev_rel, epsilon=1e-9):
        output = self.forward(self.module_input)
        frac = divide_add_epsilon(prev_rel, output, epsilon)
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

    def relevance_propagation(self, prev_rel, retain_graph, alpha=1, ):
        beta = 1 - alpha
        assert alpha - beta == 1
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

class TransposeForScores(torch.nn.Module):
    # Wonky operation in BertSelfAttention
    def forward(self, input, num_attention_heads, attention_head_size):
        new_x_shape = input.size()[:-1] + (num_attention_heads, attention_head_size)
        input = input.view(*new_x_shape)
        return input.permute(0, 2, 1, 3)
    # We just need to permute it back
    def relevance_propagation(self, prev_rel, alpha=1):
        return prev_rel.permute(0, 2, 1, 3).flatten(2)

class CloneN(torch.nn.Module):
    def forward(self, input, N):
        self.module_input = input
        self.N = N
        return (input, ) * N
    
    def relevance_propagation(self, prev_rel, alpha=1):
        frac = [divide_add_epsilon(r, self.module_input) for r in prev_rel]
        grad_times_frac = torch.autograd.grad((self.module_input, )* self.N, self.module_input, frac)[0]
        rel = self.module_input * grad_times_frac
        return rel

class Clone(torch.nn.Module):
    # Simplified clone for returning two of the same tensors
    def forward(self, input):
        self.module_input = input
        return (input, input)

    def relevance_propagation(self, prev_rel, alpha=1):
        frac = (divide_add_epsilon(prev_rel[0], self.module_input), divide_add_epsilon(prev_rel[1], self.module_input))
        grad_times_frac = torch.autograd.grad((self.module_input, )* 2, self.module_input, frac)[0]
        rel = self.module_input * grad_times_frac
        return rel

class IndexSelect(torch.nn.Module):
    # Input tensor
    # Dimension (what position the index is in [..., index, ...])
    # The index itself as a torch.Tensor
    def forward(self, inputs, dim, indices):
        self.module_input = [inputs, dim, indices] 
        return torch.index_select(inputs, dim, indices)

    def relevance_propagation(self, prev_rel, **kwargs):
        output = torch.index_select(*self.module_input)
        frac = prev_rel / output
        grad = torch.autograd.grad(output, self.module_input[0], frac, retain_graph=True)
        
        if torch.is_tensor(self.module_input[0]):
            return self.module_input[0] * grad[0]
        else:
            return (self.module_input[0][0] * grad[0], self.module_input[0][1] * grad[1])


class einsum(RelDeepTaylor):
    def forward(self, equation, *operands):
        return torch.einsum(equation, *operands)

class SoftMax(torch.nn.modules.activation.Softmax):
    def __init__(self, dim: Optional[int]) -> None:
        super().__init__(dim=dim)
        self.register_backward_hook(backward_hook)

class Sequential(torch.nn.Sequential):
    def relevance_propagation(self, prev_rel, **kwargs):
        rel = prev_rel
        for module in reversed(self._modules.values()):
            try:
                rel = module.relevance_propagation(rel, **kwargs)
            except AttributeError:
                pass
        return rel
# Main layers with custom implementation of relvance propagation,
# that is not just naïve DTD
class Linear(torch.nn.Linear, Rel):
    # Epsilon-rule
    # def relevance_propagation(self, prev_rel, **kwargs):
    #     module_output = self.forward(self.module_input)
    #     frac = divide_add_epsilon(prev_rel, module_output)
    #     frac_times_grad = frac @ self.weight
    #     return self.module_input * frac_times_grad

    # Alpha-Beta: Pancho and Daniel
    def relevance_propagation(self, prev_rel, alpha=1):
        pw = torch.clamp(self.weight, min=0)
        nw = torch.clamp(self.weight, max=0)
        px = torch.clamp(self.module_input, min=0)
        nx = torch.clamp(self.module_input, max=0)

        Z1 = F.linear(px, pw)
        Z2 = F.linear(nx, nw)
        S = divide_add_epsilon(prev_rel, Z1 + Z2)
        C1 = S @ pw
        C2 = S @ nw

        return px * C1 + nx * C2

class MatMul(Rel):
    def forward(self, inputs):
        return torch.matmul(*inputs)
    # Epsilon rule
    def relevance_propagation(self, prev_rel, alpha=1):
        X, Y = self.module_input
        module_output = self.forward((X, Y))
        frac = divide_add_epsilon(prev_rel, module_output)
        # Gradient of output w.r.t. X = Y
        grad_X = Y
        # Gradient of output w.r.t. Y = X
        grad_Y = X
        grad = torch.autograd.grad(module_output, self.module_input, frac, retain_graph=True)
        # Divide by two gotten from https://arxiv.org/pdf/1904.00605.pdf
        # return (X * torch.matmul(grad_X * frac) / 2, Y * (torch.matmul(grad_Y, frac)) / 2 )
        return (X * grad[0]) / 2, (Y * grad[1]) / 2

    # Pancho and Daniel epsilon-rule
    # def relevance_propagation(self, prev_rel, **kwargs):
    #     Q = self.module_input[0]
    #     K = self.module_input[1]

    #     module_output = self.forward(self.module_input)
    #     frac = divide_add_epsilon(prev_rel, module_output)
    #     C1 = frac @ K
    #     C2 = frac.permute(0,1,3,2) @ Q  # S is (1, 12, 197, 197). Need to transpose last 2 dims when mutliplying by Q
    #     return [Q * C1, K * C2]

class Mul(RelDeepTaylor):
    def forward(self, inputs):
        return torch.mul(*inputs)

class Add(RelEpsilon):
    
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        return torch.add(*inputs)

    def relevance_propagation(self, prev_rel, renormalize=True, alpha=1):
        X, Y = self.module_input
        module_output = self.forward(self.module_input)
        frac = divide_add_epsilon(prev_rel, module_output)
        grad = torch.autograd.grad(module_output, self.module_input, frac, retain_graph=True) # Remove if not debugging
        grad_X_times_frac = torch.ones(X.shape) * frac
        if not grad_X_times_frac.shape == X.shape:
            dims = []
            for i in range(X.dim()):
                if X.shape[i] < grad_X_times_frac.shape[i]:
                    dims.append(i)
            grad_X_times_frac = torch.sum(grad_X_times_frac, dim=dims, keepdim=True)
        grad_Y_times_frac = torch.ones(Y.shape) * frac
        if not grad_Y_times_frac.shape == Y.shape:
            dims = []
            for i in range(Y.dim()):
                if Y.shape[i] < grad_Y_times_frac.shape[i]:
                    dims.append(i)
            grad_Y_times_frac = torch.sum(grad_Y_times_frac, dim=dims, keepdim=True)
        assert (grad[0] == grad_X_times_frac).all()
        assert (grad[1] == grad_Y_times_frac).all()

        rel_X = X * grad_X_times_frac
        rel_Y = Y * grad_Y_times_frac
        if not renormalize:
            return (rel_X, rel_Y)
        
        # Tried to do it for each batch separately, but this does not work apparently
        # rel_X_sum = rel_X.sum(dim=list(range(1, rel_X.dim())))
        # rel_Y_sum = rel_Y.sum(dim=list(range(1, rel_Y.dim())))

        # X_frac = divide_add_epsilon(rel_X_sum.abs(), (rel_X_sum.abs() + rel_Y_sum.abs()) * prev_rel.sum(dim=list(range(1, prev_rel.dim()))))
        # Y_frac = divide_add_epsilon(rel_Y_sum.abs(), (rel_X_sum.abs() + rel_Y_sum.abs()) * prev_rel.sum(dim=list(range(1, prev_rel.dim()))))
        # rel_X_re = rel_X * divide_add_epsilon(X_frac.T, rel_X_sum).unsqueeze(1)
        # rel_Y_re = rel_Y * divide_add_epsilon(Y_frac.T, rel_Y_sum).unsqueeze(1)

        rel_X_sum = rel_X.sum()
        rel_Y_sum = rel_Y.sum()

        X_frac = divide_add_epsilon(rel_X_sum.abs(), rel_X_sum.abs() + rel_Y_sum.abs()) * prev_rel.sum()
        Y_frac = divide_add_epsilon(rel_Y_sum.abs(), rel_X_sum.abs() + rel_Y_sum.abs()) * prev_rel.sum()
        rel_X_re = rel_X * divide_add_epsilon(X_frac, rel_X_sum)
        rel_Y_re = rel_Y * divide_add_epsilon(Y_frac, rel_Y_sum)
        return (rel_X_re, rel_Y_re)

if __name__ == "__main__":
    module = Linear(16, 4, bias=True)
    input = torch.tensor([[1.0,2.0,4.0,5.0,1.0,2.0,4.0,5.0,1.0,2.0,4.0,5.0,1.0,2.0,4.0,5.0],[4,3,2,9,4,3,2,9,4,3,2,9,4,3,2,9]], requires_grad=True)
    output = module(input)
    module2 = Linear(4, 1)
    loss = torch.nn.functional.mse_loss(torch.nn.functional.softmax(module2(output)), torch.tensor([[1.0], [0]])).backward(retain_graph=True)
    rel = module.relevance_propagation(torch.tensor([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]))

    module = Add()
    input1 = torch.tensor([[1.0,2.0,4.0,5.0,1.0,2.0,4.0,5.0,1.0,2.0,4.0,5.0,1.0,2.0,4.0,5.0],[4,3,2,9,4,3,2,9,4,3,2,9,4,3,2,9]], requires_grad=True)
    input2 = torch.tensor([[9.0,9.0,9.0,9.0,1.0,2.0,4.0,5.0,1.0,2.0,4.0,5.0,1.0,2.0,4.0,5.0],[1,1,1,1,1,1,1,1,4,3,2,9,4,3,2,9]], requires_grad=True)
    output = module((input1, input2))
    one_hot = torch.zeros(input1.shape)
    one_hot[:, 0] = 1
    rel = module.relevance_propagation(one_hot)

    print("Done!")