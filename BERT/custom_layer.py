import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["MatMul", "Mul", "Add", "Linear", "IndexSelect", "TransposeForScores"]


def safe_divide(a, b, eps=1e-9):
    den = b.clamp(min=eps) + b.clamp(max=eps)
    den = den + den.eq(0).type(den.type()) * eps
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
    # Unused
    self.grad_input = grad_input
    self.grad_output = grad_output


class RelProp(nn.Module):
    def __init__(self):
        super(RelProp, self).__init__()
        self.register_forward_hook(forward_hook)

    def relprop(self, R, **kwargs):
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
    def relprop(self, prev_rel, **kwargs):
        return prev_rel.permute(0, 2, 1, 3).flatten(2)


class Mul(RelDeepTaylor):
    def forward(self, inputs):
        return torch.mul(*inputs)


class Add(RelProp):
    def forward(self, inputs):
        return torch.add(*inputs)

    def relprop(self, R, **kwargs):
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
        self.__setattr__("dim", dim)
        self.__setattr__("indices", indices)

        return torch.index_select(inputs, dim, indices)

    def relprop(self, R, alpha, device):
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

    def relprop_alphabeta_full(self, R, alpha=0.9, beta=0.1):
        pw = torch.clamp(self.weight, min=0)
        nw = torch.clamp(self.weight, max=0)
        px = torch.clamp(self.X, min=0)
        nx = torch.clamp(self.X, max=0)

        Z1 = F.linear(px, pw)
        Z2 = F.linear(nx, nw)
        S = safe_divide(R, Z1 + Z2)
        C1 = S @ pw
        C2 = S @ nw

        activator_relevances = px * C1 + nx * C2

        Z1 = F.linear(px, nw)
        Z2 = F.linear(nx, pw)
        S = safe_divide(R, Z1 + Z2)
        C1 = S @ pw
        C2 = S @ nw

        inhibitor_relevances = px * C1 + nx * C2

        return alpha * activator_relevances - beta * inhibitor_relevances


class MatMul(RelProp):
    def relprop(self, R, **kwargs):
        """
        LRP_eps for matrix multiplication
        """
        A = self.X[0]
        V = self.X[1]

        Z = self.forward([A, V])
        S = safe_divide(R, Z)
        """
        :dims: 
        S = (1, 12, 512, 64) = (1, h, s, Dh)
        V = (1, 12, 512, 64) = (1, h, s, Dh)
        A = (1, 12, 512, 512) = (1, h, s, s)    
        """
        C1 = S @ V.permute(0, 1, 3, 2)  # C1 = (1, 12, 512, 512)
        C2 = S.permute(0, 1, 3, 2) @ A  # C2 = (1, 12, 512, 64)
        return [A * C1 / 2, V * C2.permute(0, 1, 3, 2) / 2]

    def forward(self, X):
        assert len(X) == 2
        return torch.einsum("bhij,bhjd->bhid", X[0], X[1])
