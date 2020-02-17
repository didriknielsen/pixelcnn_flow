GPU=$1
BS=100
DOUBLE=False

EXACT_MODEL='train/exact_pixelcnn_quad/DATE_EXACT'
ELBO_MODEL_W='train/elbo_pixelcnn_quad/DATE_ELBO_W'
ELBO_MODEL_WO='train/elbo_pixelcnn_quad/DATE_ELBO_WO'

# Eval exact

CUDA_VISIBLE_DEVICES=$GPU python exact_pixelcnn_quad.py --model_path $EXACT_MODEL --batch_size $BS --double $DOUBLE

# Eval elbo w/bin cond

CUDA_VISIBLE_DEVICES=$GPU python exact_pixelcnn_quad.py --model_path $ELBO_MODEL_W --batch_size $BS --double $DOUBLE
CUDA_VISIBLE_DEVICES=$GPU python elbo_pixelcnn_quad.py --model_path $ELBO_MODEL_W --batch_size $BS --double $DOUBLE
CUDA_VISIBLE_DEVICES=$GPU python elbo_pixelcnn_quad.py --model_path $ELBO_MODEL_W --batch_size $BS --double $DOUBLE --k 10
CUDA_VISIBLE_DEVICES=$GPU python elbo_pixelcnn_quad.py --model_path $ELBO_MODEL_W --batch_size $BS --double $DOUBLE --k 100
CUDA_VISIBLE_DEVICES=$GPU python elbo_pixelcnn_quad.py --model_path $ELBO_MODEL_W --batch_size $BS --double $DOUBLE --k 1000

# Eval elbo wo/bin cond

CUDA_VISIBLE_DEVICES=$GPU python elbo_pixelcnn_quad.py --model_path $ELBO_MODEL_WO --batch_size $BS --double $DOUBLE
CUDA_VISIBLE_DEVICES=$GPU python elbo_pixelcnn_quad.py --model_path $ELBO_MODEL_WO --batch_size $BS --double $DOUBLE --k 10
CUDA_VISIBLE_DEVICES=$GPU python elbo_pixelcnn_quad.py --model_path $ELBO_MODEL_WO --batch_size $BS --double $DOUBLE --k 100
CUDA_VISIBLE_DEVICES=$GPU python elbo_pixelcnn_quad.py --model_path $ELBO_MODEL_WO --batch_size $BS --double $DOUBLE --k 1000
