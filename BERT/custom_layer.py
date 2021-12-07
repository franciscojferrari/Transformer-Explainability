import torch
import torch.nn as nn
import torch.nn.functional as F

def safe_divide(a, b):
    # TODO: Change to a new simpler safe_divice
    if a.numel() == b.numel():
        view = a.shape if a.dim() > b.dim() else b.shape
    den = b.clamp(min=1e-9) + b.clamp(max=1e-9)
    den = den + den.eq(0).type(den.type()) * 1e-9
    return a / den * b.ne(0).type(b.type())


def forward_hook(self, input, output):
    if type(input[0]) in (list, tuple):
        self.X = []
        for i in input[0]:
            x = i.detach()
            x.requires_grad = True
            self.X.append(x)
    else:
        self.X = input[0].detach()
        self.X.requires_grad = True

    self.Y = output


def backward_hook(self, grad_input, grad_output):
    self.grad_input = grad_input
    self.grad_output = grad_output


class RelProp(nn.Module):
    def __init__(self):
        super(RelProp, self).__init__()
        # if not self.training:
        self.register_forward_hook(forward_hook)

    def relprop(self, R, alpha):
        return R


class ReLU(nn.ReLU, RelProp):
    pass


class GELU(nn.GELU, RelProp):
    pass


class Softmax(nn.Softmax, RelProp):
    pass


class LayerNorm(nn.LayerNorm, RelProp):
    pass


class Dropout(nn.Dropout, RelProp):
    pass


class RelDeepTaylor(RelProp):
    def relprop(self, prev_rel, **kwargs):
        output = self.forward(self.X)
        frac = prev_rel / output
        grad = torch.autograd.grad(output, self.X, frac, retain_graph=True)
        
        if torch.is_tensor(self.module_input):
            return self.module_input * grad
        else:
            rels = []
            for i, input in enumerate(self.module_input):
                rels.append(input * grad[i])
            return rels

class TransposeForScores(torch.nn.Module):
    # Wonky operation in BertSelfAttention
    def forward(self, input, num_attention_heads, attention_head_size):
        new_x_shape = input.size()[:-1] + (num_attention_heads, attention_head_size)
        input = input.view(*new_x_shape)
        return input.permute(0, 2, 1, 3)
    # We just need to permute it back
    def relprop(self, prev_rel, alpha=1):
        return prev_rel.permute(0, 2, 1, 3).flatten(2)

class CloneN(torch.nn.Module):
    def forward(self, input, N):
        self.X = input
        self.N = N
        return (input, ) * N
    
    def relprop(self, prev_rel, alpha=1):
        frac = [safe_divide(r, self.X) for r in prev_rel]
        grad_times_frac = torch.autograd.grad((self.X, )* self.N, self.X, frac)[0]
        rel = self.X * grad_times_frac
        return rel

class Clone(torch.nn.Module):
    # Simplified clone for returning two of the same tensors
    def forward(self, input):
        self.X = input
        return (input, input)

    def relprop(self, prev_rel, alpha=1):
        frac = (safe_divide(prev_rel[0], self.X), safe_divide(prev_rel[1], self.X))
        grad_times_frac = torch.autograd.grad((self.X, )* 2, self.X, frac)[0]
        rel = self.X * grad_times_frac
        return rel

class Mul(RelDeepTaylor):
    def forward(self, inputs):
        return torch.mul(*inputs)

class Add(RelProp):
    def forward(self, inputs):
        return torch.add(*inputs)

    def relprop(self, R, alpha):
        # CHECKED: equivalent as their implementation
        Z = self.forward(self.X)
        S = safe_divide(R, Z)
        a = self.X[0] * S
        b = self.X[1] * S

        a_sum = a.sum()
        b_sum = b.sum()

        a_fact = safe_divide(a_sum.abs(), a_sum.abs() + b_sum.abs()) * R.sum()
        b_fact = safe_divide(b_sum.abs(), a_sum.abs() + b_sum.abs()) * R.sum()

        a = a * safe_divide(a_fact, a.sum())
        b = b * safe_divide(b_fact, b.sum())

        outputs = (a, b)
        return outputs


class IndexSelect(RelProp):
    # This function is used to select the relevance of only the token 0, which is the input to the MLP.
    def forward(self, inputs, dim, indices):
        self.__setattr__('dim', dim)
        self.__setattr__('indices', indices)

        return torch.index_select(inputs, dim, indices)

    def relprop(self, R, alpha, device):
        # CHECKED. Same as original implementation
        # We get R from the classification head, which is connected to final token 0
        # We have to expand the relevance R to acount for all 512 tokens, and init R as 0 for the rest of the tokens
        R_new = torch.zeros(self.X.shape).to(device)
        R_new[:, 0] = R
        return R_new


class Linear(nn.Linear, RelProp):
    def relprop(self, R, alpha, use_eps_rule=False):
        if use_eps_rule:
            return self.relprop_eps(R)
        else:
            return self.relprop_alphabeta(R)

    def relprop_eps(self, R, **kwargs):
        Z = self.forward(self.X)
        S = safe_divide(R, Z)
        C = S @ self.weight
        return self.X * C

    def relprop_alphabeta(self, R, **kwargs):
        # CHECKED: same as their implementation when alpha=1
        pw = torch.clamp(self.weight, min=0)
        nw = torch.clamp(self.weight, max=0)
        px = torch.clamp(self.X, min=0)
        nx = torch.clamp(self.X, max=0)

        Z1 = F.linear(px, pw)
        Z2 = F.linear(nx, nw)
        S = safe_divide(R, Z1 + Z2)
        C1 = S @ pw
        C2 = S @ nw

        return px * C1 + nx * C2


class MatMul1(RelProp):
    # Inheriting from RelProp we get the forward hook that sets self.X = [q, k]
    def relprop(self, R, **kwargs):
        '''
        LRP_eps for matrix multiplication
        STATUS: it works for the last block, but for rest cam_k is not the same as original repo (cam_q is OK)
            The change is only in the LAST TOKEN C2[:, :, 196, :]. The first 0-195 are exactly the same as original paper
        '''
        Q = self.X[0]
        K = self.X[1]

        Z = self.forward([Q, K])
        S = safe_divide(R, Z)
        C1 = S @ K
        # S is (1, 12, 197, 197). Need to transpose last 2 dims when mutliplying by Q
        C2 = S.permute(0, 1, 3, 2) @ Q
        return [Q * C1, K * C2]

    def forward(self, X):
        # CHECKED: this is equivalent to the forward of matmul1 in original apper
        assert len(X) == 2
        return torch.einsum('bhid,bhjd->bhij', X[0], X[1])


class MatMul2(RelProp):
    def relprop(self, R, **kwargs):
        '''
        LRP_eps for matrix multiplication
        '''
        A = self.X[0]
        V = self.X[1]

        Z = self.forward([A, V])
        S = safe_divide(R, Z)
        '''
        dims: 
        S = (1, 12, 512, 64) = (1, h, s, Dh)
        V = (1, 12, 512, 64) = (1, h, s, Dh)
        A = (1, 12, 512, 512) = (1, h, s, s)    
        '''
        C1 = S @ V.permute(0, 1, 3, 2)  # C1 = (1, 12, 512, 512)
        C2 = S.permute(0, 1, 3, 2) @ A  # C2 = (1, 12, 512, 64)
        return [A * C1 / 2, V * C2.permute(0, 1, 3, 2) / 2]

    def forward(self, X):
        assert len(X) == 2
        return torch.einsum('bhij,bhjd->bhid', X[0], X[1])