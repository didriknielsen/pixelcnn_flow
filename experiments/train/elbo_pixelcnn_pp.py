# General
import torch
import torch.nn as nn
import torch.nn.functional as F

# Setup
from setups import CategoricalImageFlowSetup

# Data
from pixelflow.data import CategoricalCIFAR10

# Model
from pixelflow.flows import UniformlyDequantizedFlow
from pixelflow.transforms import MultivariateCMOLAutoregressiveTransform2d
from pixelflow.networks.autoregressive import PixelCNNpp
from pixelflow.distributions import StandardUniform
from pixelflow.layers import LambdaLayer
from pixelflow.utils import quantize

# Optim
import torch.optim as optim

###################
## Specify setup ##
###################

setup = CategoricalImageFlowSetup()

parser = setup.get_parser()

# Model params
parser.add_argument('--nr_scales', type=int, default=3)
parser.add_argument('-q', '--nr_resnet', type=int, default=5,
                    help='Number of residual blocks per stage of the model')
parser.add_argument('-n', '--nr_filters', type=int, default=160,
                    help='Number of filters to use across the model. Higher = larger model.')
parser.add_argument('-m', '--nr_logistic_mix', type=int, default=10,
                    help='Number of logistic components in the mixture. Higher = more flexible model')
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--bin_cond', type=eval, default=True)

# Optim params
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--milestones', type=eval, default=[])
parser.add_argument('--gamma', type=float, default=0.1)

args = parser.parse_args()
setup.prepare_env(args)

##################
## Specify data ##
##################

data = CategoricalCIFAR10()

setup.register_data(data.train, data.test)

###################
## Specify model ##
###################

if args.bin_cond: lambd = lambda x: 2*quantize(x, 256) - 1
else:             lambd = lambda x: 2*(256/255)*x-1

net = nn.Sequential(LambdaLayer(lambd),
                    PixelCNNpp(nr_resnet=args.nr_resnet, nr_filters=args.nr_filters, nr_scales=args.nr_scales,
                               input_channels=3, nr_logistic_mix=args.nr_logistic_mix, dropout=args.dropout))

model = UniformlyDequantizedFlow(base_dist = StandardUniform((3,32,32,)),
                                 transforms = MultivariateCMOLAutoregressiveTransform2d(net,
                                                                                        channels=3,
                                                                                        num_mixtures=args.nr_logistic_mix,
                                                                                        num_bins=256,
                                                                                        mean_lambd=lambd))

setup.register_model(model)

#######################
## Specify optimizer ##
#######################

optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)

setup.register_optimizer(optimizer, scheduler)

###############
## Run setup ##
###############

setup.run(args)
