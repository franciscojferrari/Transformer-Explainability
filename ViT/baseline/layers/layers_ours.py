import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

__all__ = ['forward_hook', 'Add', 'ReLU', 'GELU', 'Dropout',
           'Linear', 'Conv2d',
           'safe_divide', 'Softmax', 'IndexSelect', 'LayerNorm', 'MatMul1', 'MatMul2']


def safe_divide(a, b):
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

# Reimplemented
class Add(RelProp):
    def forward(self, inputs):
        return torch.add(*inputs)

    def relprop(self, R, alpha):
        # CHECKED: closed form implementation equivalent to original 
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

        outputs = [a, b]
        return outputs

# Reimplemented
class IndexSelect(RelProp):
    # This function is used to select the relevance of only the token 0, which is the input to the MLP.
    def forward(self, inputs, dim, indices):
        self.__setattr__('dim', dim)
        self.__setattr__('indices', indices)

        return torch.index_select(inputs, dim, indices)

    def relprop(self, R, alpha, device):
        # CHECKED: closed form implementation equivalent to original 
        # We get R from the classification head, which is connected to final token 0
        # We have to expand the relevance R to acount for all 197 tokens, and init R as 0 for the rest of the tokens
        R_new = torch.zeros((1, 197, 768)).to(device)
        R_new[:, 0, :] = R
        return R_new

# Reimplemented
class Linear(nn.Linear, RelProp):
    def relprop(self, R, alpha, use_eps_rule=False):
        if use_eps_rule:
            return self.relprop_eps(R)
        else:
            return self.relprop_alphabeta(R)

    def relprop_eps(self, R):
        Z = self.forward(self.X)
        S = safe_divide(R, Z)
        C = S @ self.weight
        return self.X * C

    def relprop_alphabeta(self, R):
        # CHECKED: closed form implementation equivalent to original 
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

# Reimplemented
class MatMul1(RelProp):
    # Inheriting from RelProp we get the forward hook that sets self.X = [q, k]
    def relprop(self, R):
        '''
        LRP_eps for matrix multiplication
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

# Reimplemented
class MatMul2(RelProp):
    def relprop(self, R):
        '''
        LRP_eps for matrix multiplication
        '''
        A = self.X[0]
        V = self.X[1]

        Z = self.forward([A, V])
        S = safe_divide(R, Z)
        '''
        dims: 
        S = (1, 12, 197, 64) = (1, h, s, Dh)
        V = (1, 12, 197, 64) = (1, h, s, Dh)
        A = (1, 12, 197, 197) = (1, h, s, s)    
        '''
        C1 = S @ V.permute(0, 1, 3, 2)  # C1 = (1, 12, 197, 197)
        C2 = S.permute(0, 1, 3, 2) @ A  # C2 = (1, 12, 197, 64)
        return [A * C1, V * C2.permute(0, 1, 3, 2)]

    def forward(self, X):
        assert len(X) == 2
        return torch.einsum('bhij,bhjd->bhid', X[0], X[1])


# Not reimplemented. Used only for orignal LRP baseline 
class Conv2d(nn.Conv2d, RelProp):
    def gradprop2(self, DY, weight):
        Z = self.forward(self.X)

        output_padding = self.X.size()[2] - (
            (Z.size()[2] - 1) * self.stride[0] - 2 * self.padding[0] + self.kernel_size[0])

        return F.conv_transpose2d(
            DY, weight, stride=self.stride, padding=self.padding, output_padding=output_padding)

    def relprop(self, R, alpha):
        if self.X.shape[1] == 3:
            pw = torch.clamp(self.weight, min=0)
            nw = torch.clamp(self.weight, max=0)
            X = self.X
            L = self.X * 0 + torch.min(torch.min(torch.min(self.X, dim=1, keepdim=True)
                                       [0], dim=2, keepdim=True)[0], dim=3, keepdim=True)[0]
            H = self.X * 0 + torch.max(torch.max(torch.max(self.X, dim=1, keepdim=True)
                                       [0], dim=2, keepdim=True)[0], dim=3, keepdim=True)[0]
            Za = torch.conv2d(
                X, self.weight, bias=None, stride=self.stride, padding=self.padding) - torch.conv2d(
                L, pw, bias=None, stride=self.stride, padding=self.padding) - torch.conv2d(
                H, nw, bias=None, stride=self.stride, padding=self.padding) + 1e-9

            S = R / Za
            C = X * self.gradprop2(S, self.weight) - L * self.gradprop2(S,
                                                                        pw) - H * self.gradprop2(S, nw)
            R = C
        else:
            beta = alpha - 1
            pw = torch.clamp(self.weight, min=0)
            nw = torch.clamp(self.weight, max=0)
            px = torch.clamp(self.X, min=0)
            nx = torch.clamp(self.X, max=0)

            def f(w1, w2, x1, x2):
                Z1 = F.conv2d(x1, w1, bias=None, stride=self.stride, padding=self.padding)
                Z2 = F.conv2d(x2, w2, bias=None, stride=self.stride, padding=self.padding)
                S1 = safe_divide(R, Z1)
                S2 = safe_divide(R, Z2)
                C1 = x1 * self.gradprop(Z1, x1, S1)[0]
                C2 = x2 * self.gradprop(Z2, x2, S2)[0]
                return C1 + C2

            activator_relevances = f(pw, nw, px, nx)
            inhibitor_relevances = f(nw, pw, px, nx)

            R = alpha * activator_relevances - beta * inhibitor_relevances
        return R
