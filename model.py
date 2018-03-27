import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

class StyleTransfer(nn.Module):
    def __init__(self, args):
        super(StyleTransfer, self).__init__()
        self.args = args
        self.content_image = content_image
        self.style_image = style_image
        self.output_image = output_image
        self.vgg19 = models.vgg19(pretrained=True)
        self.optimizer = optim.Adam(
                         self.output_image.parameters(), 
                         lr = args.lr,
                         weight_decay = args.weight_decay)
        self.loss_ratio = args.loss_ratio
        self.content_layers = ['conv4']
        self.style_layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']
        

