import torch
import torch.nn as nn
from torch.autograd import Variable

class Swish(nn.Module):

  def __init__(self, B = 1.0):
    super(Swish, self).__init__()
    self.B = B

  def forward(self, x):
    Bx = x.mul(self.B)
    omega = (1 + Bx.exp()) ** (-1)
    return x.mul(omega)