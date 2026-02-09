
import torch
def linear_layer_forward(X, W, b):
    """
    Compute the forward pass of a linear (fully connected) layer.
    """
    # Write code here
    x = torch.tensor(X)
    w = torch.tensor(W)
    b_ = torch.tensor(b)
    out = x@w +b_
    return out.tolist()