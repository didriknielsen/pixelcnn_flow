# PixelCNN as a Single-Layer Flow

Official code for [Closing the Dequantization Gap: PixelCNN as a Single-Layer Flow](https://arxiv.org/abs/2002.02547) by Didrik Nielsen, Ole Winther (2020).

Code will soon be updated to include new experiments!

## Dependencies

To run the code, we used `python 3.6.8` with the following packages installed:

- `pytorch==1.2.0`
- `torchvision==0.4.0`
- `numpy==1.16.4`
- `ptable==0.9.2`


## Setup

The code used for experiments have been collected in the python package  `pixelflow`. Thus, we begin by installing this package.

In the folder containing `setup.py`, run
```
pip install --user -e .
```
The `--user` option ensures the library will only be installed for your user.  
The `-e` option makes it possible to modify the library, and modifications will be loaded on the fly.

You should now be able to use it.

## Usage

The code for training models can be found in `experiments/train`, while the code for evaluation can be found in `experiments/eval`.

### Training

The code for training models can be found in `experiments/train`.  
The `.py` files contain the experiment code and the `.sh` files contain the commands that were run:

**PixelCNN interpolation experiments:**
```
sh interpolations.sh 0
```
**Dequantization gap experiments:**
```
sh dequant_gap.sh 0
```
**Multilayer PixelCNN experiments:**
```
sh multilayer.sh 0
```

### Evaluation

The code for evaluating test performance can be found in `experiments/eval/bpd`, while the code for performing interpolations can be found in `experiments/eval/interpolations`.  
The commands used to evaluate models can be found in the `.sh` files (Note that the `MODEL_PATH` variable need to be set to the path of the pretrained models):

**PixelCNN interpolation experiments:**
```
sh interp_pixelcnn.sh 0
sh interp_pixelcnn_pp.sh 0
```

**Dequantization gap experiments:**
```
sh dequant_gap_pixelcnn.sh 0
sh dequant_gap_pixelcnn_quad.sh 0
sh dequant_gap_pixelcnn_pp.sh 0
```

**Multilayer PixelCNN experiments:**
```
sh multilayer.sh 0
```

## Acknowledgements

Parts of the code build on:
- https://github.com/bayesiains/nsf  
- https://github.com/pclucas14/pixel-cnn-pp  

Thanks to the authors of these repositories!
