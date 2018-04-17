from PIL import Image
from torch.autograd import Variable
import torchvision.transforms as transforms
import os
import numpy as np
import pdb

def load_image(filename, image_size):
  image = Image.open(filename)
  transform = transforms.Compose([
                transforms.Scale(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor()
              ])
  image = Variable(transform(image)).view(1,-1)
  return image

def save_image(image, image_size, output_path, lr, weight_decay, style_loss_weight):
  transform = transforms.ToPILImage()
  image = image.view(3, image_size, image_size)
  image = transform(image.data)
  filename = 'output_' + str(lr) + '_' + str(weight_decay) + '_' + str(style_loss_weight) + '.jpg'
  image.save(os.path.join(output_path, filename))