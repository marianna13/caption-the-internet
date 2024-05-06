# Description: Test the distributed data module
from data.utils import get_data_pipeline, write_shard_wds
import glob
import torch
import os


def test_get_data_pipeline():
    wds_path = "/p/fastdata/mmlaion/datacomp/datacomp_1B/flat/{0000000..0000270}.tar"
    dataset = get_data_pipeline(wds_path)
    assert dataset is not None
    sample = next(iter(dataset))
    assert sample is not None
    assert sample.keys() == ["image", "json", "__key__"]
    wds_path_str = "/p/fastdata/mmlaion/datacomp/datacomp_1B/flat/0000000.tar"
    dataset = get_data_pipeline(wds_path_str)
    assert dataset is not None
    sample = next(iter(dataset))
    assert sample is not None
    assert sample.keys() == ["image", "json", "__key__"]
    wds_path_list = [
        "/p/fastdata/mmlaion/datacomp/datacomp_1B/flat/0000000.tar",
        "/p/fastdata/mmlaion/datacomp/datacomp_1B/flat/0000001.tar",
    ]
    dataset = get_data_pipeline(wds_path_list)
    assert dataset is not None
    sample = next(iter(dataset))
    assert sample is not None


def test_write_shard_wds():
    base = "/p/scratch/ccstdl/marianna/temp"
    os.makedirs(base, exist_ok=True)

    source_dataset = get_data_pipeline(
        "/p/fastdata/mmlaion/datacomp/datacomp_1B/flat/0000000.tar",
        lambda x: x,
        batch_size=1,
    )
    source_dataloader = torch.utils.data.DataLoader(
        source_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x
    )
    write_shard_wds(base, source_dataloader, maxcount=1000)
    assert True
    wds_path = glob.glob(base + "/*.tar")
    assert len(wds_path) > 0
    dataset = get_data_pipeline(wds_path)
    assert dataset is not None
    sample = next(iter(dataset))
    assert sample is not None


def process_json(x):
    x["caption"] = "test"
    return x


def process_samples(batch):
    images, jsons, keys = batch

    for image, json, key in zip(images, jsons, keys):
        process_json(json)
    return batch


def test_process_samples():
    base = "/p/scratch/ccstdl/marianna/temp"
    os.makedirs(base, exist_ok=True)

    source_dataset = get_data_pipeline(
        "/p/fastdata/mmlaion/datacomp/datacomp_1B/flat/0000000.tar", process_samples
    )
    source_dataloader = torch.utils.data.DataLoader(
        source_dataset,
        shuffle=False,
        collate_fn=lambda x: x,
        num_workers=1,
        batch_size=1,
    )
    write_shard_wds(base, source_dataloader, maxcount=1000)
    assert True
    wds_path = glob.glob(base + "/*.tar")
    assert len(wds_path) > 0
    dataset = get_data_pipeline(wds_path)
    assert dataset is not None
    sample = next(iter(dataset))
    assert sample is not None
    assert sample[1][0]["caption"] == "test"


def test_split_by_worker():
    base = "/p/scratch/ccstdl/marianna/temp"
    source_dataset = get_data_pipeline(
        "/p/fastdata/mmlaion/datacomp/datacomp_1B/flat/0000000.tar",
        process_samples,
        batch_size=10,
    )
    num_workers = 10
    source_dataloader = torch.utils.data.DataLoader(
        source_dataset, num_workers=num_workers, shuffle=False, collate_fn=lambda x: x
    )

    write_shard_wds(base, source_dataloader, maxcount=1000)
    assert True
