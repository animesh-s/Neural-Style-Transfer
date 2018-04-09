import torch
import numpy as np
from torch.autograd import Variable
import model
from helpers.image import *
from helpers.gram_matrix import *
import pdb

def train(args):
  content = load_image(args.content_image)
  style = load_image(args.style_image)
  output = Variable(torch.randn(content.size()[1])).view(1, -1)
  content_loss, style_loss, total_loss = 0, 0, 0
  style_model = model.StyleTransfer(args, content, style, output)

  layer_num = 1
  for layer in style_model.vgg19:
    if type(layer) == nn.Conv2d:
      layer_name = "conv" + str(layer_name)
      if layer_name in style_model.content_layers:
        content = layer(content)
        style = layer(style)
        output = layer(output)
        content_loss = content_loss + style_model.loss(style_model.loss_ratio * content, output)
      if layer_name in style_model.style_layers:
        gram_content = gram_matrix(content)
        gram_output = gram_matrix(output)
        style_loss = style_loss + style_model.loss(style_model.loss_ratio * gram_content, gram_output)
    if type(layer) == nn.ReLU:
      layer_num = layer_num + 1

  total_loss = content_loss + style_loss
