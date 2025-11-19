import argparse
import glob
import os

from zenodo_client import update_zenodo

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Uploads all files from the current directory into a Zenodo Record")

    parser.add_argument("zenodo_id",
                        help="Zenodo Deposition Identifier (the numbers in the Zenodo URL)",
                        type=str)

    args = parser.parse_args()

    zenodo_id = args.zenodo_id

    paths = glob.glob("*")

    total_size = 0
    for path in paths:
        total_size += os.path.getsize(path)
    total_size_gb = total_size * 1e-9

    if total_size_gb > 50:
        raise Exception(
            f"Total file size {total_size_gb} GB is bigger than the Zenodo max size of 50 GB.")

    print(f"Uploading {total_size_gb} GB to Zenodo Record.")
    update_zenodo(zenodo_id, paths, publish=False)
