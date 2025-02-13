#!/bin/bash -x
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --job-name=batch_inference
#SBATCH --account=transfernetx
#SBATCH --partition=booster
#SBATCH --threads-per-core=1
#SBATCH --time=0:15:00
#SBATCH --exclusive
#SBATCH --output=output/%x_%j.out


export HF_HOME=HF_HOMEcache
export OPEN_CLIP_CACHE_DIR=open_clip_cache

export CATPTION_THE_INTERNET_DIR=caption-the-internet

OPEN_CLIP_PATH="/p/project1/laionize/marianna/open-clip/open_clip_all_at_once/src"
export PYTHONPATH=${OPEN_CLIP_PATH}:${PYTHONPATH}

NUM_GPUS=4
export SLURM_GPUS_PER_NODE=$NUM_GPUS
export CUDA_VISIBLE_DEVICES=0,1,2,3

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)


head_node_i="${head_node}i"

head_node_ip="$(nslookup "$head_node_i" | grep -oP '(?<=Address: ).*')"
export head_node_ip=$head_node_ip
echo "Head node: $head_node_ip"

port=20156
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

NUM_GPUS=4

echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" --gres=gpu:4 \
    ray start --head --node-ip-address="$head_node_ip" --port=$port --num-gpus $NUM_GPUS \
    --num-cpus ${SLURM_CPUS_PER_TASK}  --block &

sleep 10

worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= worker_num; i++)); do
    node=${nodes_array[$i]}
    node_i="${node}i"
    echo "Starting WORKER $i at $node"
    this_node_ip="$(nslookup "$node_i" | grep -oP '(?<=Address: ).*')"
    srun --nodes=1 --ntasks=1 -w "$node" --gres=gpu:4 \
        ray start --address "$ip_head" \
        --node-ip-address="$this_node_ip"  \
        --num-cpus ${SLURM_CPUS_PER_TASK} --num-gpus $NUM_GPUS --block &
    sleep 10
done

export RAY_ADDRESS="$head_node_ip:$port"

sleep 30
ray status

cd ${CATPTION_THE_INTERNET_DIR}

MODEL="deepseek-ai/deepseek-vl2-tiny"
MODEL_PATH="/p/data1/mmlaion/marianna/models/${MODEL}"
DATA_PATH="/p/data1/mmlaion/laion_bvd/extracted_images_test/000-part-00001-c4778e33-5c8f-4681-aa83-d8837fc6294a-c000.snappy.parquet/"
OUTPUT_PATH="/p/data1/mmlaion/laion_bvd_recap/${MODEL}/"
PROMPT="write a very short caption for the image"
META_KEY="info.json"
IMG_KEY="png"
TP=1
DP=4
BATCH_SIZE=32
NUM_WORKERS=4
BACKEND="vllm"
MAX_TOKENS=15

CMD="
python -u src/batch_inference.py \
    --model_path ${MODEL_PATH} \
    --data_path ${DATA_PATH} \
    --output_dir ${OUTPUT_PATH} \
    --prompt \"${PROMPT}\" \
    --meta-key ${META_KEY} \
    --batch_size ${BATCH_SIZE} \
    --num_workers ${NUM_WORKERS} \
    --max_tokens ${MAX_TOKENS} \
    --tp ${TP} \
    --dp ${DP} \
    --backend ${BACKEND} \
    --img_key ${IMG_KEY} \
    --compute_clip_score \
    "
bash -c "${CMD}"
