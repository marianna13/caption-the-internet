from vllm import LLM, SamplingParams
from open_clip_models import OpenClipModel
import fire
from visualization import plot_grid
import pyarrow as pa
import ray
import shutil
import os
import glob
import torch
from tqdm import tqdm
from clip_score_utils import ClipScoreProcessor
from prompts import PROMPT_MAP
from extra_args import EXTRA_ARGS_MAP
from vllm.distributed import destroy_model_parallel, destroy_distributed_environment
import gc
import contextlib
from data_routines.data import get_data_loader
from data_routines.writer import ParquetSampleWriter


def cleanup_vllm(llm):
    destroy_model_parallel()
    destroy_distributed_environment()
    del llm.llm_engine.model_executor
    del llm
    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def get_done_shards(tmp_dir):
    done_shards = []
    for f in os.listdir(tmp_dir):
        if f.endswith(".feather"):
            done_shards.append(f.split(".")[0])
    return done_shards


def get_clipscores(
    data_path,
    parquet_path,
    clip_model="openai/clip-vit-base-patch32",
    meta_key="info.json",
    img_key="image",
    output_dir="output",
    shard_id=0,
    batch_size=32,
):
    print(f"Computing CLIP scores with model: {clip_model}")

    clip_score = ClipScoreProcessor(clip_model)
    dataloader = get_data_loader(
        data_path, batch_size=batch_size, num_workers=4, meta_key=meta_key
    )

    metadata_pq = pa.parquet.read_table(parquet_path).to_pandas()
    print(metadata_pq)

    images, captions = [], []

    for batch_id, batch in tqdm(enumerate(dataloader)):
        if batch_id > 0:
            break

        for i, b in enumerate(batch):
            row = metadata_pq.iloc[i]
            row = row.to_dict()
            img = b["image"]
            caption = row["txt"]
            captions.append(caption)
            images.append(img)

    n_col = 4
    n_row = (
        len(images) // n_col if len(images) % n_col == 0 else len(images) // n_col + 1
    )
    plot_grid(
        images,
        n_col,
        n_row,
        captions,
        orig_captions=None,
        fig_path=os.path.join(output_dir, f"{shard_id}.png"),
    )

    clip_scores = clip_score(captions, images)
    avg_score = sum(clip_scores) / len(clip_scores)
    print(f"Average CLIP score: {avg_score:.2f}")


@ray.remote(num_gpus=4, num_cpus=24)
def process_shard(
    shard_id,
    data_path,
    model_path,
    output_dir,
    tp,
    rank=0,
    img_key="image",
    meta_key="info.json",
    prompt="",
    sampling_params=None,
    batch_size=32,
    num_workers=4,
    backend="vllm",
    compute_clip_score=False,
    debug=False,
):
    model = get_model(model_path, rank, backend, tp)
    dataloader = get_data_loader(
        data_path,
        batch_size=batch_size,
        num_workers=num_workers,
        meta_key=meta_key,
    )
    schema = pa.schema(
        [
            pa.field("txt", pa.string()),
            pa.field("prompt", pa.string()),
            pa.field("url", pa.string()),
            pa.field("key", pa.string()),
        ]
    )
    writer = ParquetSampleWriter(
        shard_id=shard_id,
        output_folder=output_dir,
        save_caption=True,
        oom_shard_count=0,
        schema=schema,
        encode_format="jpg",
    )

    for batch_id, batch in tqdm(enumerate(dataloader)):
        if debug and batch_id > 0:
            break
        print(batch[0].keys())
        metas = [b["json"] for b in batch]
        imgs = [b["image"] for b in batch]
        urls = [b["__url__"] for b in batch]
        keys = [b["__key__"] for b in batch]

        if backend == "vllm":
            sampling_params = SamplingParams(**sampling_params)
            prompt_func = PROMPT_MAP["/".join(model_path.split("/")[-2:])]
            prompt = prompt_func(prompt)

        inputs = [
            {
                "prompt": prompt,
                "multi_modal_data": {
                    "image": img,
                },
            }
            for img in imgs
        ]
        captions = model.generate(
            inputs,
            sampling_params=sampling_params,
        )

        torch.cuda.empty_cache()
        captions = [c.outputs[0].text for c in captions]
        for i in range(len(imgs)):
            meta_data = metas[i]
            meta_data["url"] = urls[i]
            meta_data["prompt"] = prompt
            writer.write(
                key=keys[i],
                caption=captions[i],
                meta=meta_data,
            )

    writer.close()

    if backend == "vllm":
        cleanup_vllm(model)

    if compute_clip_score:
        get_clipscores(
            data_path,
            writer.pq_file,
            meta_key=meta_key,
            img_key=img_key,
            output_dir=output_dir,
            shard_id=shard_id,
        )

    print(f"Processed {batch_id} batches")
    print(os.path.join(output_dir, f"{shard_id}.png"))

    tmp_dir = os.path.join(output_dir, "_tmp")

    with open(os.path.join(tmp_dir, f"{shard_id}.feather"), "w") as f:
        f.write("done")


def get_model(model_path, rank, backend="vllm", tensor_parallel_size=1):
    """
    Get the model based on the backend

    Args:
        model_path (str): Path to the model
        rank (int): Rank of the process
        backend (str): Backend to use (vllm or open_clip)
    """

    if backend == "vllm":
        print("/".join(model_path.split("/")[-2:]))
        extra_args = EXTRA_ARGS_MAP.get("/".join(model_path.split("/")[-2:]), {})
        print(f"Extra args: {extra_args}")
        model = LLM(
            model=model_path,
            trust_remote_code=True,
            max_model_len=4096,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=0.98,
            max_num_seqs=1,
            **extra_args,
        )
    elif backend == "open_clip":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = OpenClipModel(model_path, device=device)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    return model


def batch_inference(
    model_path,
    data_path,
    prompt,
    max_tokens=128,
    temperature=0.6,
    tp=1,
    batch_size=32,
    num_workers=4,
    output_dir="output",
    img_key="image",
    meta_key="info.json",
    backend="vllm",
    compute_clip_score=False,
    debug=False,
):
    print(model_path)

    sampling_params = {
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    shards = glob.glob(f"{data_path}/*.tar")
    shards = shards[:4]

    tmp_dir = os.path.join(output_dir, "_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    done_shards = get_done_shards(tmp_dir)
    cuda_devices = torch.cuda.device_count()
    print(f"Found {cuda_devices} cuda devices")
    ray.init(
        address="auto",
        # num_cpus=24,
        # num_gpus=cuda_devices,
        include_dashboard=False,
    )

    futures = [
        process_shard.options(num_gpus=4).remote(
            shard_id=shard_id,
            data_path=shards[shard_id],
            model_path=model_path,
            output_dir=output_dir,
            tp=tp,
            prompt=prompt,
            img_key=img_key,
            meta_key=meta_key,
            sampling_params=sampling_params,
            backend=backend,
            compute_clip_score=compute_clip_score,
            batch_size=batch_size,
            num_workers=num_workers,
            debug=debug,
        )
        for shard_id in range(len(shards))
        if f"{shard_id}" not in done_shards
    ]

    ray.get(futures)

    ray.shutdown()
    shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    fire.Fire(batch_inference)
