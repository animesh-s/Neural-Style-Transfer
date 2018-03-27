import os
import argparse
import torch

parser = argparse.ArgumentParser(description='Neural Style Transfer')

# Image parameters
parser.add_argument('-content-image', type=str, default='images/content/hoover_tower.jpg', help='Content image')
parser.add_argument('-style-image', type=str, default='images/style/starry_night.jpg', help='Style image')
parser.add_argument('-output-image', type=str, default='output', help='Output image')

#Learning parameters
parser.add_argument('-pooling', default='avg', type=str, choices=['max', 'avg'], help='Pooling scheme')
parser.add_argument('-weight-decay', type=float, default=0.00019, help='Weight decay to use for training')
parser.add_argument('-lr', type=float, default=0.00001, help='Learning rate to use for training')
parser.add_argument('-optimizer', type=str, default='SGD', help='Loss function optimizer (SGD or Adam) [default: SGD]')
parser.add_argument('-epochs', type=int, default=5, help='Number of epochs for train [default: 5]')
parser.add_argument('-iterations', type=int, default=None, help='Number of iterations for train [default: None]')
parser.add_argument('-batch-size', type=int, default=32, help='Number of examples in a batch [default:32]')
parser.add_argument('-log-interval', type=int, default=6480,  help='Steps to wait before logging training status [default: 1]')
parser.add_argument('-save-dir', type=str, default='images/output', help='Where to save the snapshots')

# CNN parameters
parser.add_argument('-kernel-sizes', type=str, default='2,3,4,5', help='Comma-separated kernel size to use for convolution')
parser.add_argument('-dropout', type=float, default=0.1, help='Probability for dropout [default: 0.1]')
parser.add_argument('-kernel-number', type=int, default=100, help='Number of each kind of kernel')

# Device
parser.add_argument('-no-cuda', action='store_true', default=False, help='Disable the gpu' )
parser.add_argument('-cv', action='store_true', default=False, help='CV or full run')

args = parser.parse_args()
args.cuda = (not args.no_cuda) and torch.cuda.is_available(); del args.no_cuda
args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
if not os.path.isdir(args.save_dir): os.makedirs(args.save_dir)

print("\nParameters:")
for attr, value in sorted(args.__dict__.items()):
    print("\t{}={}".format(attr.upper(), value))
