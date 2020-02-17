# Model
from pixelflow.flows import UniformlyDequantizedFlow
from pixelflow.transforms import MultivariateCMOLAutoregressiveTransform2d
from pixelflow.networks.autoregressive import PixelCNNpp
from pixelflow.distributions import StandardUniform
from pixelflow.layers import LambdaLayer
from pixelflow.utils import quantize


def create_model_fn(args):

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

    return model

####################
## Evaluate model ##
####################

from _eval_elbo import eval_elbo

eval_elbo(create_model_fn=create_model_fn)
