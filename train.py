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
  content_loss, style_loss = 0, 0
  style_model = model.StyleTransfer(args, content, style, output)