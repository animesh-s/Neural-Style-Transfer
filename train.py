import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import model
from helpers.image import *
from helpers.gram_matrix import *
import pdb

def train(args):
  content = load_image(args.content_image, args.image_size)
  style = load_image(args.style_image, args.image_size)
  output = Variable(torch.randn(content.size()[1]), requires_grad = True).view(1, -1)
  style_model = model.StyleTransfer(args, content, style, output)

  for epoch in range(args.iterations):
    content = style_model.content_image.clone().view(1, 3, args.image_size, args.image_size)
    style = style_model.style_image.clone().view(1, 3, args.image_size, args.image_size)
    output = style_model.output_image.clone().view(1, 3, args.image_size, args.image_size)
    print output
    print 'Epoch: ' + str(epoch)
    content_loss, style_loss, loss = 0, 0, 0
    layer_num = 1
    style_model.optimizer.zero_grad()
    for layer in style_model.vgg19:
      if type(layer) == nn.ReLU:
        layer = nn.ReLU(inplace=False)
        layer_num = layer_num + 1
      content = layer(content)
      style = layer(style)
      output = layer(output)
      if type(layer) == nn.Conv2d:
        layer_name = "conv" + str(layer_num)
        if layer_name in style_model.content_layers:
          content_loss = content_loss + style_model.loss(style_model.content_loss_weight * output, style_model.content_loss_weight * content.detach())
        if layer_name in style_model.style_layers:
          gram_style = gram_matrix(style)
          gram_output = gram_matrix(output)
          style_loss = style_loss + style_model.loss(style_model.style_loss_weight * gram_output, style_model.style_loss_weight * gram_style.detach())
    loss = content_loss + style_loss
    print 'Loss: ' + str(loss.data[0])
    loss.backward()
    style_model.optimizer.step()

  save_image(style_model.output_image, args.image_size, args.save_dir)
