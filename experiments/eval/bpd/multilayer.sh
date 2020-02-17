GPU=$1
BS=100
DOUBLE=False

MODEL1='train/exact_pixelcnn/DATE1'
MODEL2='train/exact_deep_pixelcnn_drop/DATE2'
MODEL4='train/exact_deep_pixelcnn_drop/DATE4'
MODEL4D='train/exact_deep_pixelcnn_drop/DATE4D'

MODEL1Q='train/exact_pixelcnn_quad/DATE1Q'
MODEL2Q='train/exact_deep_pixelcnn_quad_drop/DATE2Q'
MODEL4Q='train/exact_deep_pixelcnn_quad_drop/DATE4Q'
MODEL4QD='train/exact_deep_pixelcnn_quad_drop/DATE4QD'

# Eval PixelCNN

CUDA_VISIBLE_DEVICES=$GPU python exact_pixelcnn.py --model_path $MODEL1 --batch_size $BS --double $DOUBLE
CUDA_VISIBLE_DEVICES=$GPU python exact_deep_pixelcnn_drop.py --model_path $MODEL2 --batch_size $BS --double $DOUBLE
CUDA_VISIBLE_DEVICES=$GPU python exact_deep_pixelcnn_drop.py --model_path $MODEL4 --batch_size $BS --double $DOUBLE
CUDA_VISIBLE_DEVICES=$GPU python exact_deep_pixelcnn_drop.py --model_path $MODEL4D --batch_size $BS --double $DOUBLE

# Eval QuadPixelCNN

CUDA_VISIBLE_DEVICES=$GPU python exact_pixelcnn_quad.py --model_path $MODEL1Q --batch_size $BS --double $DOUBLE
CUDA_VISIBLE_DEVICES=$GPU python exact_deep_pixelcnn_quad_drop.py --model_path $MODEL2Q --batch_size $BS --double $DOUBLE
CUDA_VISIBLE_DEVICES=$GPU python exact_deep_pixelcnn_quad_drop.py --model_path $MODEL4Q --batch_size $BS --double $DOUBLE
CUDA_VISIBLE_DEVICES=$GPU python exact_deep_pixelcnn_quad_drop.py --model_path $MODEL4QD --batch_size $BS --double $DOUBLE

