import torch
import numpy as np
from torch.autograd import Variable
import model
from helpers.image import *
import pdb

def train(args):
  content = load_image(args.content_image)
  style = load_image(args.style_image)
  content_loss, style_loss = 0, 0
  save_image(content, args.save_dir)

