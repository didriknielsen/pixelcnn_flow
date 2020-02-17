GPU=$1



# PixelCNN

python exact_pixelcnn.py --num_epochs 500 --batch_size 16 --lr 3e-4 --gamma 0.5 --milestones [250,300,350,400,450] --test_every 1 --checkpoint_every 10 --num_bins 256 --gpus [$GPU]
python exact_deep_pixelcnn_drop.py --num_epochs 500 --batch_size 16 --lr 3e-4 --gamma 0.5 --milestones [250,300,350,400,450] --test_every 1 --checkpoint_every 10 --num_bins 256 --num_flows 2 --gpus [$GPU]
python exact_deep_pixelcnn_drop.py --num_epochs 500 --batch_size 16 --lr 3e-4 --gamma 0.5 --milestones [250,300,350,400,450] --test_every 1 --checkpoint_every 10 --num_bins 256 --num_flows 4 --gpus [$GPU]
python exact_deep_pixelcnn_drop.py --num_epochs 500 --batch_size 16 --lr 3e-4 --gamma 0.5 --milestones [250,300,350,400,450] --test_every 1 --checkpoint_every 10 --num_bins 256 --num_flows 4 --dropout 0.2 --gpus [$GPU]



# PixelCNN (Quad)

python exact_pixelcnn_quad.py --num_epochs 500 --batch_size 16 --lr 3e-4 --gamma 0.5 --milestones [250,300,350,400,450] --test_every 1 --checkpoint_every 10 --num_bins 16 --gpus [$GPU]
python exact_deep_pixelcnn_quad_drop.py --num_epochs 500 --batch_size 16 --lr 3e-4 --gamma 0.5 --milestones [250,300,350,400,450] --test_every 1 --checkpoint_every 10 --num_bins 16 --num_flows 2 --gpus [$GPU]
python exact_deep_pixelcnn_quad_drop.py --num_epochs 500 --batch_size 16 --lr 3e-4 --gamma 0.5 --milestones [250,300,350,400,450] --test_every 1 --checkpoint_every 10 --num_bins 16 --num_flows 4 --gpus [$GPU]
python exact_deep_pixelcnn_quad_drop.py --num_epochs 500 --batch_size 16 --lr 3e-4 --gamma 0.5 --milestones [250,300,350,400,450] --test_every 1 --checkpoint_every 10 --num_bins 16 --num_flows 4 --dropout 0.2 --gpus [$GPU]

