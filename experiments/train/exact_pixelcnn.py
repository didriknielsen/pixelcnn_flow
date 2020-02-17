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
from pixelflow.transforms.subset import LinearSplineAutoregressiveSubsetTransform2d
from pixelflow.networks.autoregressive import PixelCNN

# Optim
import torch.optim as optim

###################
## Specify setup ##
###################

setup = CategoricalImageFlowSetup()

parser = setup.get_parser()

# Model params
parser.add_argument('--num_bins', type=int, default=256)
parser.add_argument('--filters', type=int, default=128)
parser.add_argument('--num_blocks', type=int, default=15)
parser.add_argument('--kernel_size', type=int, default=3)
parser.add_argument('--kernel_size_in', type=int, default=7)
parser.add_argument('--output_filters', type=int, default=1024)

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

model = AutoregressiveSubsetFlow2d(base_shape = (3,32,32,),
                                   transforms = [
                                   LinearSplineAutoregressiveSubsetTransform2d(PixelCNN(3, num_params=args.num_bins,
                                                                                        num_blocks=args.num_blocks,
                                                                                        filters=args.filters,
                                                                                        kernel_size=args.kernel_size,
                                                                                        kernel_size_in=args.kernel_size_in,
                                                                                        output_filters=args.output_filters,
                                                                                        init_transforms=lambda x: 2*x-1), num_bins=args.num_bins),
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
