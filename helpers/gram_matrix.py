import torch

def gram_matrix(input):
  a, b, c, d = input.size()
  features = input.view(a * b, c * d)
  G = torch.mm(features, features.t())
  return G.div(a * b * c * d)