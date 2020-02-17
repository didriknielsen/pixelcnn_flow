# Model
from pixelflow.flows import AutoregressiveSubsetFlow2d
from pixelflow.transforms.subset import LinearSplineAutoregressiveSubsetTransform2d
from pixelflow.networks.autoregressive import DropPixelCNN


def create_model_fn(args):

    transforms = []
    for i in range(args.num_flows):

        transforms += [LinearSplineAutoregressiveSubsetTransform2d(DropPixelCNN(3,
                                                                                num_params=args.num_bins[i],
                                                                                num_blocks=args.num_blocks[i],
                                                                                filters=args.filters[i],
                                                                                kernel_size=args.kernel_size[i],
                                                                                kernel_size_in=args.kernel_size_in[i],
                                                                                output_filters=args.output_filters[i],
                                                                                dropout=args.dropout[i],
                                                                                init_transforms=lambda x: 2 * x - 1), num_bins=args.num_bins[i])]

    model = AutoregressiveSubsetFlow2d(base_shape = (3,32,32,),
                                       transforms = transforms)

    return model

####################
## Evaluate model ##
####################

from _eval_exact import eval_exact

eval_exact(create_model_fn=create_model_fn)
