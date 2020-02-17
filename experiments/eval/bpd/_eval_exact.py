import os
import argparse
import pickle
import torch

# Data
from pixelflow.data import CategoricalCIFAR10
from pixelflow.data.utils import DEFAULT_PATH
LOG_FOLDER = os.path.join(os.path.dirname(DEFAULT_PATH), 'experiments/log')
CHECK_FOLDER = os.path.join(os.path.dirname(DEFAULT_PATH), 'experiments/check')


def dataset_loglik_bpd(model, data_loader, device, double):
    with torch.no_grad():
        bpd = 0.0
        count = 0
        for i, (batch, _) in enumerate(data_loader):
            if double: batch = batch.double()
            bpd += model.loglik_bpd(batch.to(device)).cpu().item() * len(batch)
            count += len(batch)
            print('{}/{}'.format(i+1, len(data_loader)), bpd/count, end='\r')
    return bpd / count


def eval_exact(create_model_fn):

    parser = argparse.ArgumentParser()

    # Training args
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--test_set', type=eval, default=True)
    parser.add_argument('--double', type=eval, default=False)

    eval_args = parser.parse_args()

    batch_size = eval_args.batch_size
    model_log = os.path.join(LOG_FOLDER, eval_args.model_path)
    model_check = os.path.join(CHECK_FOLDER, eval_args.model_path)

    with open('{}/args.pickle'.format(model_log), 'rb') as f:
        args = pickle.load(f)

    ##################
    ## Specify data ##
    ##################

    torch.manual_seed(0)

    data = CategoricalCIFAR10()

    if eval_args.test_set:
        data_loader = torch.utils.data.DataLoader(data.test, batch_size=batch_size)
    else:
        data_loader = torch.utils.data.DataLoader(data.train, batch_size=batch_size)
    test_str = 'test' if eval_args.test_set else 'train'

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
    double_str = '_double' if eval_args.double else ''

    ####################
    ## Compute loglik ##
    ####################

    bpd = dataset_loglik_bpd(model, data_loader, device=eval_args.device, double=eval_args.double)
    print('Done, bpd: {}'.format(bpd))
    fname = '{}/exact_loglik_bpd_{}{}.txt'.format(model_log, test_str, double_str)
    with open(fname, "w") as f:
        f.write(str(bpd))
