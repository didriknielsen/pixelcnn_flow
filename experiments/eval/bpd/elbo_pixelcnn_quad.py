# Model
from pixelflow.flows import UniformlyDequantizedFlow
from pixelflow.transforms import QuadraticSplineAutoregressiveTransform2d
from pixelflow.networks.autoregressive import PixelCNN
from pixelflow.distributions import StandardUniform
from pixelflow.utils import quantize


def create_model_fn(args):

    if args.bin_cond: lambd = lambda x: 2*quantize(x, 256) - 1
    else:             lambd = lambda x: 2*x-1

    net = PixelCNN(3, num_params=2*args.num_bins+1,
                   num_blocks=args.num_blocks,
                   filters=args.filters,
                   kernel_size=args.kernel_size,
                   kernel_size_in=args.kernel_size_in,
                   output_filters=args.output_filters,
                   init_transforms=lambd)

    model = UniformlyDequantizedFlow(base_dist = StandardUniform((3,32,32,)),
                                     transforms = QuadraticSplineAutoregressiveTransform2d(net, num_bins=args.num_bins))

    return model

####################
## Evaluate model ##
####################

from _eval_elbo import eval_elbo

eval_elbo(create_model_fn=create_model_fn)
