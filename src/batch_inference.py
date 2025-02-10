from vllm import LLM, SamplingParams
from open_clip_models import OpenClipModel
import fire
from data import get_data_loader
from writer import ParquetSampleWriter
from viz import plot_grid
import pyarrow as pa
import ray
import shutil
import os
import glob
import torch
from tqdm import tqdm
from clip_score_utils import ClipScoreProcessor


def get_done_shards(tmp_dir):
    done_shards = []
    for f in os.listdir(tmp_dir):
        if f.endswith(".feather"):
            done_shards.append(f.split(".")[0])
    return done_shards


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

    img_key = "image"
    meta_key = "json"

    for batch_id, batch in tqdm(enumerate(dataloader)):
        if batch_id > 0:
            break
        print(batch[0].keys())
        metas = [b[meta_key] for b in batch]
        imgs = [b[img_key] for b in batch]
        urls = [b["__url__"] for b in batch]
        keys = [b["__key__"] for b in batch]

        inputs = [
            {
                "prompt": prompt,
                "multi_modal_data": {
                    "image": img,
                },
            }
            for img in imgs
        ]
        if backend == "vllm":
            sampling_params = SamplingParams(**sampling_params)
        captions = model.generate(
            inputs,
            sampling_params=sampling_params,
        )
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

        if compute_clip_score:
            new_captions = []
            clip_score = ClipScoreProcessor("openai/clip-vit-base-patch32")
            scores = clip_score(captions, imgs)
            print(scores)
            avg_score = sum(scores) / len(scores)
            print(f"Average CLIP score: {avg_score:.2f}")
            for i in range(len(imgs)):
                new_captions.append(f"{captions[i]} [CLIP score: {scores[i]:.2f}]")
            # for i in range(len(imgs)):
            #     score = clip_score(captions[i], imgs[i])
            #     new_captions.append(
            #         f"{captions[i]} [CLIP score: {score:.2f}]"
            #     )
            captions = new_captions

        print(f"Processed {batch_id} batches")
        print(os.path.join(output_dir, f"{shard_id}.png"))
        plot_grid(
            imgs,
            4,
            8,
            captions,
            orig_captions=None,
            fig_path=os.path.join(output_dir, f"{shard_id}.png"),
        )

        del imgs
    writer.close()

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
        model = LLM(
            model=model_path,
            trust_remote_code=True,
            max_model_len=4096,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=0.98,
            max_num_seqs=1,
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
):
    print(model_path)

    sampling_params = {
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    # sampling_params = SamplingParams(**sampling_params)

    shards = glob.glob(f"{data_path}/*.tar")
    shards = shards[:1]

    tmp_dir = os.path.join(output_dir, "_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    done_shards = get_done_shards(tmp_dir)
    cuda_devices = torch.cuda.device_count()
    print(f"Found {cuda_devices} cuda devices")
    ray.init(
        address="auto",
        # num_cpus=24,
        # num_gpus=cuda_devices,
        # include_dashboard=False,
    )

    futures = [
        process_shard.remote(
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
        )
        for shard_id in range(len(shards))
        if f"{shard_id}" not in done_shards
    ]

    ray.get(futures)

    ray.shutdown()
    shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    fire.Fire(batch_inference)
