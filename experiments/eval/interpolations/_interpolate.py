import os
import math
import pickle
import argparse
import numpy as np
import torch
import torchvision.utils as vutils
from torch.distributions import Normal

# Data
from pixelflow.data import CategoricalCIFAR10
from pixelflow.data.utils import DEFAULT_PATH
LOG_FOLDER = os.path.join(os.path.dirname(DEFAULT_PATH), 'experiments/log')
CHECK_FOLDER = os.path.join(os.path.dirname(DEFAULT_PATH), 'experiments/check')


def interpolate(create_model_fn, idx_list):

    parser = argparse.ArgumentParser()

    # Training args
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--row_length', type=int, default=9)
    parser.add_argument('--double', type=eval, default=False)
    parser.add_argument('--clamp', type=eval, default=False)

    eval_args = parser.parse_args()

    model_log = os.path.join(LOG_FOLDER, eval_args.model_path)
    model_check = os.path.join(CHECK_FOLDER, eval_args.model_path)

    with open('{}/args.pickle'.format(model_log), 'rb') as f:
        args = pickle.load(f)

    torch.manual_seed(0)

    u = torch.rand(3,32,32).to(eval_args.device)
    if eval_args.double: u = u.double()

    ###############
    ## Load data ##
    ###############

    data = CategoricalCIFAR10()

    ################
    ## Load model ##
    ################

    model = create_model_fn(args)

    # Load pre-trained weights
    weights = torch.load('{}/model.pt'.format(model_check), map_location='cpu')
    model.load_state_dict(weights, strict=False)
    model = model.to(eval_args.device)
    model = model.eval()
    if eval_args.double: model = model.double()

    ############################
    ## Perform interpolations ##
    ############################

    gaussian = Normal(0,1)

    idxs = idx_list[eval_args.start:eval_args.end]

    with torch.no_grad():
        data1, data2 = [], []
        batch_idxs = []
        for n, (i1, i2) in enumerate(idxs):

            data1.append(data.test[i1][0].unsqueeze(0))
            data2.append(data.test[i2][0].unsqueeze(0))
            batch_idxs.append((i1,i2))

            if (n+1) % eval_args.batch_size == 0 or (n+1)==len(idxs):

                data1 = torch.cat(data1,dim=0)
                data2 = torch.cat(data2,dim=0)

                print("Matching pairs", (n+1)-eval_args.batch_size, "-", n+1, "/", len(idxs))

                if eval_args.double:
                    data1 = data1.double()
                    data2 = data2.double()
                double_str = '_double' if eval_args.double else ''

                z_lower1, z_upper1 = model.forward_transform(data1.to(eval_args.device))
                z_lower2, z_upper2 = model.forward_transform(data2.to(eval_args.device))

                z1 = z_lower1 + (z_upper1 - z_lower1) * u
                z2 = z_lower2 + (z_upper2 - z_lower2) * u

                # Move latent to Gaussian space
                g1 = gaussian.icdf(z1)
                g2 = gaussian.icdf(z2)
                g1[g1==-math.inf] = -1e9
                g1[g1==math.inf] = 1e9
                g2[g2==-math.inf] = -1e9
                g2[g2==math.inf] = 1e9

                # Interpolation in Gaussian space:
                ws = [(w / (math.sqrt(w**2 + (1-w)**2)), (1-w) / (math.sqrt(w**2 + (1-w)**2))) for w in np.linspace(0,1,eval_args.row_length)]
                zw = torch.cat([gaussian.cdf(w[0] * g1 + w[1] * g2) for w in ws], dim=0)
                xw = model.inverse_transform(zw, clamp=eval_args.clamp).cpu().float()/255
                xw = xw.reshape(eval_args.row_length, len(batch_idxs), *xw.shape[1:])
                for i, (i1, i2) in enumerate(batch_idxs):
                    vutils.save_image(xw[:,i], '{}/i_{}_{}_l_{}{}.png'.format(model_log, i1, i2, eval_args.row_length, double_str), nrow=eval_args.row_length, padding=2)
                print("Stored interpolations")

                data1, data2 = [], []
                batch_idxs = []
