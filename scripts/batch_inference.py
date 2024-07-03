from benchmark_captioners import load_hf_model, get_captioner_func
import argparse
import webdataset as wds
import json
import glob
import os
import braceexpand
from accelerate import PartialState
import logging
import torch
from typing import Callable
from data.utils import get_data_pipeline, write_shard_wds
import torch.distributed
import torch.utils.data


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logging.warning(f"Handling webdataset error ({repr(exn)}). Ignoring.")
    return True


class BatchProcessor:
    def __init__(
        self,
        captioner_func: Callable,
        captioner: str,
        tokenizer: str,
        image_processor: str,
        args: argparse.Namespace,
        config: dict,
    ):
        self.captioner_func = captioner_func
        self.captioner = captioner
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.args = args
        self.config = config

    def __call__(self, batch):
        # print(batch)

        images, jsons, keys, url = batch[0]

        captions, _ = self.captioner_func(
            model_name=self.args.captioner,
            model=self.captioner,
            tokenizer=self.tokenizer,
            images=images,
            args=self.args,
            config=self.config,
            image_processor=self.image_processor,
            max_new_tokens=self.args.max_new_tokens,
        )

        for js, caption in zip(jsons, captions):
            js["original_caption"] = js["caption"]
            js["caption"] = caption.replace('"', "")
            js["url"] = url
        return batch

def get_last_processed_shard(output_dir):
    '''
    Each shard is assigned id (0,1, …, N)
    Each subshard has the name {shard_id}_{subshard_id}.tar.
    After each shard is fully processed save {shard_id}_stats.json
    When starting, check which id was processed the last (if not applicable assign start_id=0) by checking existing stats.json.
    '''
    stats_jsons = glob.glob(f"{output_dir}/*_stats.json")
    done_shard_ids = [int(stats_json.split("/")[-1].split("_")[0]) for stats_json in stats_jsons]
    done_shard_ids.sort()
    return done_shard_ids[-1]

def get_done_urls(output_dir):
    '''
    Each shard is assigned id (0,1, …, N)
    Each subshard has the name {shard_id}_{subshard_id}.tar.
    After each shard is fully processed save {shard_id}_stats.json
    When starting, check which id was processed the last (if not applicable assign start_id=0) by checking existing stats.json.
    '''
    stats_txt = glob.glob(f"{output_dir}/**/stats.txt", recursive=True)
    done_urls = set()
    for stats in stats_txt:
        with open(stats, "r") as f:
            done_urls.update(f.read().split("\n"))

    return done_urls

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--captioner", type=str, required=True, help="Name of the captioner to use"
    )
    parser.add_argument(
        "--batch_size", type=int, required=True, help="Batch size to use"
    )
    parser.add_argument(
        "--num_gpus", type=int, required=True, help="Number of GPUs to use"
    )
    parser.add_argument(
        "--wds_path", type=str, required=True, help="Path to webdataset"
    )
    parser.add_argument(
        "--quant", choices=[4, 8], type=int, default=None, help="quantization bits"
    )
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--config", type=str, default="config.json")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--limit", type=int, default=None)

    slurm_node_id = int(os.environ.get("SLURM_NODEID"))

    args = parser.parse_args()
    rank = args.rank
    with open(args.config, "r") as f:
        config = json.load(f)

    MODELS_DICT = config["models"]
    # captioner = MODELS_DICT[args.captioner]
    captioner = args.captioner

    distributed_state = PartialState()

    args.device_map = distributed_state.device

    args.rank = int(os.environ.get("LOCAL_RANK", -1))
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        group = torch.distributed.group.WORLD
        rank = torch.distributed.get_rank(group=group)
        world_size = torch.distributed.get_world_size(group=group)
        print("world_size:", world_size, "rank:", rank)

    print(torch.utils.data.get_worker_info())
    print("rank:", args.rank)

    captioner, tokenizer, image_processor = load_hf_model(
        args.captioner, args, MODELS_DICT[captioner]["path"]
    )

    captioner_func = get_captioner_func(args.captioner)
    if '..' in args.wds_path:
        args.wds_path = list(braceexpand.braceexpand(args.wds_path))
    elif os.path.isdir(args.wds_path):
        args.wds_path = glob.glob(f"{args.wds_path}/**/*.tar", recursive=True)
    else:
        args.wds_path = [args.wds_path]

    limit = args.limit
    print("limit:", limit)
    if limit:
        args.wds_path = args.wds_path[:limit]
    done_urls = get_done_urls(args.output_dir)

    args.wds_path = [wds_url for wds_url in args.wds_path if wds_url not in done_urls]

    args.wds_path.sort()
    print(len(args.wds_path))

    slurm_node_id = int(os.environ.get("SLURM_NODEID"))
    print("slurm_node_id:", slurm_node_id)

    args.prompt = "Write a short 100-word caption for the image:"

    output_dir = f"{args.output_dir}/{slurm_node_id:02d}_{args.rank:02d}"
    os.makedirs(output_dir, exist_ok=True)

    # shard_id = get_last_processed_shard(output_dir)

    # output_path = f"{output_dir}/{shard_id:08d}.tar"
    batch_processor = BatchProcessor(
        captioner_func=captioner_func,
        captioner=captioner,
        tokenizer=tokenizer,
        image_processor=image_processor,
        args=args,
        config=MODELS_DICT[args.captioner],
    )


    dataset = get_data_pipeline(args.wds_path, batch_size=args.batch_size)
    # dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x)
    dataloader = wds.WebLoader(dataset, shuffle=False, collate_fn=lambda x: x)
    write_shard_wds(
        output_dir, dataloader, process_samples=batch_processor, maxcount=20000
    )
    print("Done")


if __name__ == "__main__":
    main()
