GPU=$1



## PixelCNN

# Exact
python exact_pixelcnn.py --num_epochs 500 --batch_size 16 --lr 3e-4 --gamma 0.5 --milestones [250,300,350,400,450] --test_every 1 --checkpoint_every 10 --num_bins 256 --gpus [$GPU]
# ELBO, w / bin conditioning
python elbo_pixelcnn.py --num_epochs 500 --batch_size 16 --lr 3e-4 --gamma 0.5 --milestones [250,300,350,400,450] --test_every 1 --checkpoint_every 10 --num_bins 256 --bin_cond True --gpus [$GPU]
# ELBO, wo / bin conditioning
python elbo_pixelcnn.py --num_epochs 500 --batch_size 16 --lr 3e-4 --gamma 0.5 --milestones [250,300,350,400,450] --test_every 1 --checkpoint_every 10 --num_bins 256 --bin_cond False --gpus [$GPU]




## PixelCNN (Quad)

# Exact
python exact_pixelcnn_quad.py --num_epochs 500 --batch_size 16 --lr 3e-4 --gamma 0.5 --milestones [250,300,350,400,450] --test_every 1 --checkpoint_every 10 --num_bins 16 --gpus [$GPU]
# ELBO, w / bin conditioning
python elbo_pixelcnn_quad.py --num_epochs 500 --batch_size 16 --lr 3e-4 --gamma 0.5 --milestones [250,300,350,400,450] --test_every 1 --checkpoint_every 10 --num_bins 16 --bin_cond True --gpus [$GPU]
# ELBO, wo / bin conditioning
python elbo_pixelcnn_quad.py --num_epochs 500 --batch_size 16 --lr 3e-4 --gamma 0.5 --milestones [250,300,350,400,450] --test_every 1 --checkpoint_every 10 --num_bins 16 --bin_cond False --gpus [$GPU]






## PixelCNN++

# Exact
python exact_pixelcnn_pp.py --num_epochs 500 --batch_size 16 --lr 3e-4 --gamma 0.5 --milestones [250,300,350,400,450] --test_every 1 --checkpoint_every 10 --gpus [$GPU]
# ELBO, w / bin conditioning
python elbo_pixelcnn_pp.py --num_epochs 500 --batch_size 16 --lr 3e-4 --gamma 0.5 --milestones [250,300,350,400,450] --test_every 1 --checkpoint_every 10 --bin_cond True --gpus [$GPU]
# ELBO, wo / bin conditioning
python elbo_pixelcnn_pp.py --num_epochs 500 --batch_size 16 --lr 3e-4 --gamma 0.5 --milestones [250,300,350,400,450] --test_every 1 --checkpoint_every 10 --bin_cond False --gpus [$GPU]

