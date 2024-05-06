import webdataset as wds
import os
import json
from typing import Callable, Union, Optional
import time
import torch
import braceexpand
import logging
from itertools import islice


_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000


def my_split_by_worker(urls):
    wi = torch.utils.data.get_worker_info()
    if wi is None:
        return urls
    else:
        return urls[wi.id :: wi.num_workers]


def my_split_by_node(src, group=None):
    node_id, node_count = (
        torch.distributed.get_rank(),
        torch.distributed.get_world_size(),
    )
    print("pytorch id", node_id, "pytorch world size", node_count)
    if node_count > 1:
        yield from islice(src, node_id, None, node_count)
    else:
        yield from src


def split_by_slurm_node(src, group=None):
    """Split the input sequence by PyTorch distributed rank."""
    rank = int(os.environ["SLURM_NODEID"])
    world_size = int(os.environ["SLURM_NNODES"])
    print("slurm node id", rank, "slurm num nodes", world_size)
    if world_size > 1:
        yield from islice(src, rank, None, world_size)
    else:
        yield from src


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logging.warning(f"Handling webdataset error ({repr(exn)}). Ignoring.")
    return True


def get_data_pipeline(
    wds_path: Union[str, list],
    process_samples: Optional[Callable[[dict], dict]] = None,
    batch_size: int = 1,
):
    """Get a WebDataset data pipeline."""
    if isinstance(wds_path, str):
        wds_path = list(braceexpand.braceexpand(wds_path))
    pipeline = [wds.SimpleShardList(wds_path)]

    pipeline.extend(
        [
            split_by_slurm_node,
            wds.split_by_node,
            wds.split_by_worker,
            wds.tarfile_to_samples(handler=log_and_continue),
            wds.split_by_worker,
            wds.shuffle(
                bufsize=_SAMPLE_SHUFFLE_SIZE,
                initial=_SAMPLE_SHUFFLE_INITIAL,
            ),
            wds.decode("pil", handler=log_and_continue),
            wds.rename(image="jpg;webp;jpeg;png", handler=log_and_continue),
            # wds.to_tuple("image", "json", "__key__"),
        ]
    )

    pipeline.extend(
        [wds.to_tuple("image", "json", "__key__", "__url__"), wds.batched(batch_size)]
    )

    if process_samples is not None:
        pipeline.append(wds.map(process_samples))
    dataset = wds.DataPipeline(*pipeline)
    return dataset


def write_shard_wds(
    base: str,
    source_dataloader: torch.utils.data.DataLoader,
    maxsize: int = 3e9,
    maxcount: int = 100000,
    start: int = 0,
    num_processes: int = 1,
    device: torch.device = None,
    process_samples: Optional[Callable[[dict], dict]] = None,
):
    """Write a shard of samples to a WebDataset tar file."""
    pattern = os.path.join(base, "%08d.tar")
    stats_json = os.path.join(base, "stats_00.json")

    with wds.ShardWriter(
        pattern, maxsize=int(maxsize), maxcount=int(maxcount), start_shard=start
    ) as sink:
        s = time.time()
        stats = {
            "count": 0,
            "id": [],
            "original_caption": [],
            "url": [],
            "img_size": [],
        }
        for i, batch in enumerate(source_dataloader):
            # print(len(batch), len(batch[0]), len(batch[0][0]))

            if process_samples is not None:
                batch = process_samples(batch)

            for images, jsons, keys, urls in batch:
                for image, json_data, key, url in zip(images, jsons, keys, urls):
                    sample = {
                        "jpg": image,
                        "json": json_data,
                        "__key__": key,
                        "txt": json_data["caption"],
                    }
                    try:
                        sink.write(sample)
                    except Exception as e:
                        logging.warning(f"Error writing sample: {e}")
                        continue
                    del sample
                    size_of_pil_image_bytes = len(image.tobytes())
                    stats["count"] += 1
                    stats["id"].append(json_data["uid"])
                    stats["original_caption"].append(json_data["caption"])
                    stats["url"].append(url)
                    stats["img_size"].append(size_of_pil_image_bytes)

            with open(stats_json, "w") as f:
                json.dump(stats, f, indent=4)
            stats_json = os.path.join(base, f"stats_{i+1:02d}.json")
            stats = {
                "count": 0,
                "id": [],
                "original_caption": [],
                "url": [],
                "img_size": [],
            }
        logging.info(f"Processed in: {time.time() - s}.", end="\r")
