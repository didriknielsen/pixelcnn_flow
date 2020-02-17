GPU=$1
BS=4
DOUBLE=True
CLAMP=True

MODEL_PATH=train/exact_pixelcnn_pp/DATE_EXACT

CUDA_VISIBLE_DEVICES=$GPU python pixelcnn_pp.py --model_path $MODEL_PATH --batch_size $BS --double $DOUBLE --clamp $CLAMP
