# General
import torch
import torch.nn as nn
import torch.nn.functional as F

# Setup
from setups import CategoricalImageFlowSetup

# Data
from pixelflow.data import CategoricalCIFAR10

# Model
from pixelflow.flows import AutoregressiveSubsetFlow2d
from pixelflow.transforms.subset import MultivariateMOLAutoregressiveSubsetTransform2d
from pixelflow.networks.autoregressive import PixelCNNpp
from pixelflow.layers import LambdaLayer

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

net = nn.Sequential(LambdaLayer(lambda x: 2 * (256 / 255) * x - 1),
                    PixelCNNpp(nr_resnet=args.nr_resnet, nr_filters=args.nr_filters, nr_scales=args.nr_scales,
                               input_channels=3, nr_logistic_mix=args.nr_logistic_mix, dropout=args.dropout))

model = AutoregressiveSubsetFlow2d(base_shape = (3,32,32,),
                                   transforms = [
                                   MultivariateMOLAutoregressiveSubsetTransform2d(net,
                                                                                  channels=3,
                                                                                  num_mixtures=args.nr_logistic_mix,
                                                                                  num_bins=256,
                                                                                  mean_lambd=lambda x: 2*(x*256/255)-1)
                                   ])

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
