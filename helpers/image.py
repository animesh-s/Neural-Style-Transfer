from PIL import Image
from torch.autograd import Variable
import torchvision.transforms as transforms
import os
import numpy as np
import pdb

image_size = 128
channels = 3

def load_image(filename):
  image = Image.open(filename)
  transform = transforms.Compose([
                transforms.Scale(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor()
              ])
  image = Variable(transform(image)).view(1,-1)
  return image

def save_image(image, output_path):
  transform = transforms.ToPILImage()
  image = image.view(channels, image_size, image_size)
  image = transform(image.data)
  image.save(os.path.join(output_path, 'output.jpg'))