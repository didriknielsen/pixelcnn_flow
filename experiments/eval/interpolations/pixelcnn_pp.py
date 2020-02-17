# Model
from pixelflow.flows import AutoregressiveSubsetFlow2d
from pixelflow.transforms.subset import MultivariateMOLAutoregressiveSubsetTransform2d
from pixelflow.networks.autoregressive import PixelCNNpp
from pixelflow.layers import LambdaLayer


def create_model_fn(args):

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

    return model

####################
## Evaluate model ##
####################

from _interpolate import interpolate

idx_list = [(1890,2338),
            (3402,314),
            (4592,8710),
            (5480,6405)]

interpolate(create_model_fn=create_model_fn,
            idx_list=idx_list)
