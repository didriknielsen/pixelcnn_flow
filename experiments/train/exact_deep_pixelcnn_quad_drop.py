# General
import torch
import torch.nn as nn
import torch.nn.functional as F

# Setup
from setups import CategoricalImageFlowSetup
from setups.argparse import prep_int, prep_float, prep_str, prep_bool

# Data
from pixelflow.data import CategoricalCIFAR10

# Model
from pixelflow.flows import AutoregressiveSubsetFlow2d
from pixelflow.transforms.subset import QuadraticSplineAutoregressiveSubsetTransform2d
from pixelflow.networks.autoregressive import DropPixelCNN

# Optim
import torch.optim as optim

###################
## Specify setup ##
###################

setup = CategoricalImageFlowSetup()

parser = setup.get_parser()

# Model params
parser.add_argument('--num_flows', type=int, default=2)
parser.add_argument('--num_bins', type=int, default=16)
parser.add_argument('--filters', type=int, default=128)
parser.add_argument('--num_blocks', type=int, default=15)
parser.add_argument('--kernel_size', type=int, default=3)
parser.add_argument('--kernel_size_in', type=int, default=7)
parser.add_argument('--output_filters', type=int, default=1024)
parser.add_argument('--dropout', type=eval, default=0.0)

# Optim params
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--milestones', type=eval, default=[])
parser.add_argument('--gamma', type=float, default=0.1)

args = parser.parse_args()
setup.prepare_env(args)

#######################
## Process arguments ##
#######################

args.num_bins = prep_int(args.num_bins, args.num_flows)
args.filters = prep_int(args.filters, args.num_flows)
args.num_blocks = prep_int(args.num_blocks, args.num_flows)
args.kernel_size = prep_int(args.kernel_size, args.num_flows)
args.kernel_size_in = prep_int(args.kernel_size_in, args.num_flows)
args.output_filters = prep_int(args.output_filters, args.num_flows)
args.dropout = prep_float(args.dropout, args.num_flows)

##################
## Specify data ##
##################

data = CategoricalCIFAR10()

setup.register_data(data.train, data.test)

###################
## Specify model ##
###################

transforms = []
for i in range(args.num_flows):

    transforms += [QuadraticSplineAutoregressiveSubsetTransform2d(DropPixelCNN(3,
                                                                               num_params=2*args.num_bins[i]+1,
                                                                               num_blocks=args.num_blocks[i],
                                                                               filters=args.filters[i],
                                                                               kernel_size=args.kernel_size[i],
                                                                               kernel_size_in=args.kernel_size_in[i],
                                                                               output_filters=args.output_filters[i],
                                                                               dropout=args.dropout[i],
                                                                               init_transforms=lambda x: 2 * x - 1), num_bins=args.num_bins[i])]

model = AutoregressiveSubsetFlow2d(base_shape = (3,32,32,),
                                   transforms = transforms)

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
