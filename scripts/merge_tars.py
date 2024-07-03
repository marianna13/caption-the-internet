import os
import argparse
import glob


def check_if_file_empty(file_path):
    return os.stat(file_path).st_size == 0


def rename_file(file_path, new_file_path):
    os.rename(file_path, new_file_path)

def merge_tars(input_dir, output_dir, only_nested=False):
    tar_files = glob.glob(f'{input_dir}/**/*.tar', recursive=False) if only_nested else glob.glob(f'{input_dir}/*.tar')
    for i, tar_file in enumerate(tar_files):
        if check_if_file_empty(tar_file):
            print(f"File {tar_file} is empty. Skipping...")
            continue
        print(f"Renaming {tar_file} to {output_dir}/{i:06d}.tar...")
        rename_file(tar_file, f'{output_dir}/{i:06d}.tar')
    print("All tar files merged successfully.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--only_nested', action='store_true')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    merge_tars(args.input_dir, args.output_dir, args.only_nested)

    
if __name__ == "__main__":
    main()
