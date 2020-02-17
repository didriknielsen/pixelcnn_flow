GPU=$1



# ImageNet32

python imagenet32_exact_deep_pixelcnn.py --num_epochs 500 --batch_size 16 --lr 3e-4 --gamma 0.5 --milestones [250,300,350,400,450] --test_every 1 --checkpoint_every 10 --num_flows 1 --gpus [$GPU]
python imagenet32_exact_deep_pixelcnn.py --num_epochs 500 --batch_size 16 --lr 3e-4 --gamma 0.5 --milestones [250,300,350,400,450] --test_every 1 --checkpoint_every 10 --num_flows 2 --gpus [$GPU]
python imagenet32_exact_deep_pixelcnn_quad.py --num_epochs 500 --batch_size 16 --lr 3e-4 --gamma 0.5 --milestones [250,300,350,400,450] --test_every 1 --checkpoint_every 10 --num_flows 2 --gpus [$GPU]

# ImageNet64

python imagenet64_exact_deep_pixelcnn.py --num_epochs 500 --batch_size 16 --lr 3e-4 --gamma 0.5 --milestones [250,300,350,400,450] --test_every 1 --checkpoint_every 10 --num_flows 1 --gpus [$GPU]
python imagenet64_exact_deep_pixelcnn.py --num_epochs 500 --batch_size 16 --lr 3e-4 --gamma 0.5 --milestones [250,300,350,400,450] --test_every 1 --checkpoint_every 10 --num_flows 2 --gpus [$GPU]
python imagenet64_exact_deep_pixelcnn_quad.py --num_epochs 500 --batch_size 16 --lr 3e-4 --gamma 0.5 --milestones [250,300,350,400,450] --test_every 1 --checkpoint_every 10 --num_flows 2 --gpus [$GPU]



