from scripts.merge_tars import merge_tars
import pytest
import os
import os.path
import glob
import shutil


@pytest.mark.parametrize(
    "tmp_path",
    ["/tmp"]
)
def test_merge_tars(tmp_path:str):
    input_dir = f"{tmp_path}/input"
    output_dir = f"{tmp_path}/output"
    
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    num_files = 10

    # Create a tar files
    for i in range(num_files):
        with open(f"{input_dir}/file{i}.tar", "w") as f:
            f.write("test")

    num_tar_files = len(glob.glob(f"{input_dir}/*.tar"))
    num_nested_tar_files = len(glob.glob(f"{input_dir}/**/*.tar", recursive=False))

    assert num_tar_files == num_files
    assert num_nested_tar_files == 0

    initial_size_of_input_dir_bytes = sum(os.path.getsize(f"{input_dir}/file{i}.tar") for i in range(num_files))

    merge_tars(input_dir, output_dir, only_nested=False)

    for i in range(num_files):
        assert not os.path.exists(f"{input_dir}/file{i}.tar")
        assert os.path.exists(f"{output_dir}/{i:06d}.tar")
        assert not os.stat(f"{output_dir}/{i:06d}.tar").st_size == 0
        assert os.stat(f"{output_dir}/{i:06d}.tar").st_size == 4
    
    assert sum(os.path.getsize(f"{output_dir}/{i:06d}.tar") for i in range(num_files)) == initial_size_of_input_dir_bytes
    assert len(os.listdir(output_dir)) == num_files
    assert len(os.listdir(input_dir)) == 0
    shutil.rmtree(input_dir)
    shutil.rmtree(output_dir)
    

@pytest.mark.parametrize(
    "tmp_path",
    ["/tmp"]
)
def test_merge_tars_nested_dir(tmp_path):
    input_dir = f"{tmp_path}/input"
    output_dir = f"{tmp_path}/output"
    
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    num_files = 10

    # Create a tar files
    for i in range(num_files):
        os.makedirs(f"{input_dir}/dir{i}", exist_ok=True)
        with open(f"{input_dir}/dir{i}/file{i}.tar", "w") as f:
            f.write("test")
    

    initial_size_of_input_dir_bytes = sum(os.path.getsize(f"{input_dir}/dir{i}/file{i}.tar") for i in range(num_files))

    num_tar_files = len(glob.glob(f"{input_dir}/*.tar"))
    num_nested_tar_files = len(glob.glob(f"{input_dir}/**/*.tar", recursive=False))

    assert num_tar_files == 0
    assert num_nested_tar_files == num_files

    merge_tars(input_dir, output_dir, only_nested=True)

    for i in range(num_files):
        assert not os.path.exists(f"{input_dir}/dir{i}/file{i}.tar")
        assert os.path.exists(f"{output_dir}/{i:06d}.tar")
        assert not os.stat(f"{output_dir}/{i:06d}.tar").st_size == 0
        assert os.stat(f"{output_dir}/{i:06d}.tar").st_size == 4
    
    assert sum(os.path.getsize(f"{output_dir}/{i:06d}.tar") for i in range(num_files)) == initial_size_of_input_dir_bytes
    shutil.rmtree(input_dir)
    shutil.rmtree(output_dir)