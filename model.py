import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torchvision.models as models

torch.manual_seed(1)

class StyleTransfer(nn.Module):
    def __init__(self, args):
        super(StyleTransfer, self).__init__()
        self.args = args
        self.content_image = content_image
        self.style_image = style_image
        self.output_image = output_image
        self.vgg19 = models.vgg19(pretrained=True)
        self.optimizer = optim.Adam(
                         self.output_image.parameters(), lr = args.lr,
                         weight_decay = args.weight_decay)
        

