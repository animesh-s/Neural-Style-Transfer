import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

class StyleTransfer(nn.Module):
  def __init__(self, args, content_image, style_image, output_image):
    super(StyleTransfer, self).__init__()
    self.args = args
    self.content_image = content_image
    self.style_image = style_image
    self.output_image = nn.Parameter(output_image.data)
    self.vgg19 = models.vgg19(pretrained=True).features
    self.optimizer = optim.Adam(
                       [self.output_image], 
                       lr = args.lr,
                       weight_decay = args.weight_decay
                      )
    self.content_loss_weight = args.content_loss_weight
    self.style_loss_weight = args.style_loss_weight
    self.content_layers = [21]
    self.style_layers = [0, 5, 10, 19, 28]
    self.loss = nn.MSELoss()