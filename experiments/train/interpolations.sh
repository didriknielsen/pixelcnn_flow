GPU=$1

# PixelCNN

python exact_pixelcnn.py --num_epochs 500 --batch_size 16 --lr 3e-4 --gamma 0.5 --milestones [250,300,350,400,450] --test_every 1 --checkpoint_every 10 --num_bins 256 --gpus [$GPU]

# PixelCNN++

python exact_pixelcnn_pp.py --num_epochs 500 --batch_size 16 --lr 3e-4 --gamma 0.5 --milestones [250,300,350,400,450] --test_every 1 --checkpoint_every 10 --num_bins 256 --gpus [$GPU]

