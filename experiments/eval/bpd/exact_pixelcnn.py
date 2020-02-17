# Model
from pixelflow.flows import AutoregressiveSubsetFlow2d
from pixelflow.transforms.subset import LinearSplineAutoregressiveSubsetTransform2d
from pixelflow.networks.autoregressive import PixelCNN


def create_model_fn(args):

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

    return model

####################
## Evaluate model ##
####################

from _eval_exact import eval_exact

eval_exact(create_model_fn=create_model_fn)
