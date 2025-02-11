import torch
from torch import nn, Tensor


class PromptEmbed(nn.Module):
    def __init__(self, tensor, regularizer: float = 3):
        """A simple class to update the embeddings during training time
        The regularizer controls the margins of the norm of the updated embeddings, the higher the regularizer, the more lenient the controls"""
        super(PromptEmbed, self).__init__()
        self.tensor = nn.Parameter(tensor)
        self.initial_l2 = self.get_norm().detach()
        self.regularizer = regularizer

    def get_norm(self):
        l2_norms = torch.norm(self.tensor, p=2, dim=2)
        average_l2_norm = l2_norms.mean()
        return average_l2_norm

    def forward(self):
        regularizer_loss = ((self.get_norm() - self.initial_l2) / self.regularizer).square()
        return self.tensor.data, regularizer_loss
