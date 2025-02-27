import torch
import torch.nn as nn


# -------------------------
# Global network parameters
# -------------------------

# Those parameters serve as quantization factors
# - Quantization is used to transform evaluation values to appropriate range
SCALE = 400
QA = 255
QB = 64


# ---------------------------
# Custom activation functions
# ---------------------------

# CReLU
class CReLU(nn.Module):
    def __init__(self, M: float = 1.0):
        super(CReLU, self).__init__()
        self.M = M
    
    def forward(self, x):
        return torch.clamp(x, 0.0, self.M)

# SCReLU is a modified version of ReLU, with the following formula: f(x) = clamp(x, 0, M)^2
# - An additional parameter M defines maximal range for clamp operations
# - By default M = 1
class SCReLU(nn.Module):
    def __init__(self):
        super(SCReLU, self).__init__()

    def forward(self, x):
        return torch.clamp(x, 0.0, 1.0) ** 2
    
# Standard logistic function with normalization
class Sigmoid(nn.Module):
    def __init__(self, Qa, Qb, Scale, use_Qa=True):
        super(Sigmoid, self).__init__()
        self.register_buffer("Qa", torch.tensor(Qa))
        self.register_buffer("Qb", torch.tensor(Qb))
        self.register_buffer("Scale", torch.tensor(Scale))
        self.use_Qa = use_Qa

    def forward(self, x):
        if self.use_Qa:
            x = x / self.Qa
        x = (self.Scale * x) / (self.Qa * self.Qa)
        return torch.sigmoid(x)