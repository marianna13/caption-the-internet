#

CONFIG=$1

OUTPUT_DIR=/tmp/caption_the_internet/benchmark
BATCH_SIZE=32
QUANT=4
DATA_DIR=/tmp/caption_the_internet/data
IMAGES_DIR=/tmp/caption_the_internet/images
MODEL_NAME=llava


# Run the benchmark
python3 -m caption_the_internet.benchmark \
    --config $CONFIG \
    --output_dir $OUTPUT_DIR \
    --batch_size $BATCH_SIZE \
    --quant $QUANT \
    --plot \
    --data_dir $DATA_DIR \
    --images_dir $IMAGES_DIR \
    --model_name $MODEL_NAME \
