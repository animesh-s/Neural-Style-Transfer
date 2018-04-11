import torch

def gram_matrix(input):
  batch = input.size()[0]
  channels = input.size()[1]
  height = input.size()[2]
  width = input.size()[3]
  input = input.view(batch * channels, height * width)
  G = torch.mm(input, input.t())
  G = G.div(batch * channels * height * width)
  return G