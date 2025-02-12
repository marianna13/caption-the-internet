import ray
import yaml
import fire
import dataclasses
import glob
import pyarrow as pa
import os
from typing import List
from .data import get_data_loader
from .filters import FILTERS
from .writer import ParquetSampleWriter
from visualization import plot_grid


@dataclasses.dataclass
class DataConfig:
    data_path: str
    batch_size: int
    num_workers: int
    meta_key: str
    img_key: str
    filter_functions: List[dict]
    output_dir: str
    txt_key: str = None
    make_visualization: bool = False


@ray.remote
def process_shard(
    shard_id: int,
    shard_path: str,
    output_dir: str,
    config: DataConfig,
    debug: bool = False,
):
    schema = [
        ("key", pa.string()),
        ("txt", pa.string()),
        ("video_file", pa.string()),
    ]

    # schema.append(("txt", pa.string()))

    schema = pa.schema(schema)
    txt_key = config.txt_key
    writer = ParquetSampleWriter(
        shard_id=shard_id,
        output_folder=output_dir,
        save_caption=True,
        oom_shard_count=0,
        schema=schema,
        encode_format="jpg",
    )

    filters = []
    for ff in config.filter_functions:
        filter_type = ff["type"]
        filter = FILTERS[filter_type]
        filter_config = filter["config"](**ff)
        filter_fn = filter["filter"](filter_config)
        filters.append(filter_fn)

    dataloader = get_data_loader(
        data_path=shard_path,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        meta_key=config.meta_key,
        filter_functions=filters,
    )

    for batch_id, batch in enumerate(dataloader):
        if debug and batch_id > 0:
            break
        metas = [b["json"] for b in batch]
        keys = [b["__key__"] for b in batch]
        if config.make_visualization and batch_id == 0:
            imgs = [b["image"] for b in batch]
            rows = len(batch) // 4
            cols = 4
            captions = keys
            fig_path = os.path.join(output_dir, f"shard_{shard_id}.png")
            plot_grid(
                imgs,
                rows,
                cols,
                captions,
                orig_captions=None,
                figsize=(20, 12),
                fig_path=fig_path,
            )
            print(f"Saved visualization to {fig_path}")

        for key, meta in zip(keys, metas):
            writer.write(key, meta[txt_key] if txt_key else None, meta)

        # writer.flush()

    writer.close()

    print(f"Finished processing shard {shard_id}")
    print(f"Output written to {output_dir}")


def main(
    config_path: str = None,
    debug: bool = False,
):
    assert config_path is not None, "Please provide a config file"
    config = yaml.safe_load(open(config_path))
    data_config = DataConfig(**config)

    shards = glob.glob(os.path.join(data_config.data_path, "*.tar"))[:1]
    print(f"Processing {len(shards)} shards")

    output_dir = data_config.output_dir
    os.makedirs(output_dir, exist_ok=True)

    ray.init(address="auto")
    ret = []
    for shard_id, shard_path in enumerate(shards):
        ret.append(
            process_shard.remote(
                shard_id, shard_path, data_config.output_dir, data_config, debug
            )
        )
        # process_shard(shard_id, shard_path, output_dir, data_config)

    ray.get(ret)


if __name__ == "__main__":
    fire.Fire(main)
