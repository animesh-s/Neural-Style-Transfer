import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import model
from helpers.image import *
from helpers.gram_matrix import *
import pdb

def train(args):
  for num_model in range(args.num_models):
    content = load_image(args.content_image, args.image_size)
    style = load_image(args.style_image, args.image_size)
    output = load_image(args.content_image, args.image_size)
    args.lr = 10**np.random.uniform(-5, -1)
    args.weight_decay = 10**np.random.uniform(-5, 1)
    args.style_loss_weight = 10**np.random.uniform(3.5, 4.5)
    print '\nModel: ' + str(num_model + 1) + '    LR: ' + str(args.lr) + '    Weight Decay: ' + \
          str(args.weight_decay) + '    Style Loss Weight: ' + str(args.style_loss_weight)
    style_model = model.StyleTransfer(args, content, style, output)
    for epoch in range(args.iterations):
      content = style_model.content_image.clone().view(1, 3, args.image_size, args.image_size)
      style = style_model.style_image.clone().view(1, 3, args.image_size, args.image_size)
      output = style_model.output_image.clone().view(1, 3, args.image_size, args.image_size)
      output.data.clamp_(0, 1)
      print 'Epoch: ' + str(epoch)
      content_loss, style_loss, loss = 0, 0, 0
      layer_num = 0
      style_model.optimizer.zero_grad()
      for layer in style_model.vgg19:
        if type(layer) == nn.ReLU:
          layer = nn.ReLU(inplace=False)
        content = layer(content)
        style = layer(style)
        output = layer(output)
        if type(layer) == nn.Conv2d:
          if layer_num in style_model.content_layers:
            content_loss = content_loss + style_model.loss(style_model.content_loss_weight * output, style_model.content_loss_weight * content.detach())
          if layer_num in style_model.style_layers:
            gram_style = gram_matrix(style)
            gram_output = gram_matrix(output)
            style_loss = style_loss + style_model.loss(style_model.style_loss_weight * gram_output, style_model.style_loss_weight * gram_style.detach())
        layer_num = layer_num + 1
      loss = content_loss + style_loss
      print 'Loss: ' + str(loss.data[0])
      loss.backward()
      style_model.optimizer.step()
    style_model.output_image.data.clamp_(0, 1)
    save_image(style_model.output_image, args.image_size, args.save_dir, args.lr, args.weight_decay, args.style_loss_weight)
