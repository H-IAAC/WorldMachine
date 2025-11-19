import argparse
import glob
import multiprocessing as mp
import os
import pathlib
import zipfile
from concurrent.futures import Future, ProcessPoolExecutor, as_completed

import tqdm


def create_zip(file_name: str,
               base_directory: str | pathlib.Path,
               export_directory: str = "",
               include_extensions: list[str] | None = None,
               include_dirs: list[str] | None = None,
               exclude_dirs: list[str] | None = None) -> None:

    base_directory = pathlib.Path(base_directory)

    export_path = os.path.join(export_directory, file_name+".zip")
    with zipfile.ZipFile(export_path,
                         mode="w",
                         compression=zipfile.ZIP_LZMA,  # zipfile.ZIP_DEFLATED,
                         compresslevel=9) as archive:
        for file_path in tqdm.tqdm(base_directory.rglob("*"), position=int(os.environ["TQDM_POSITION"])):
            file_path: pathlib.Path

            # Include dirs
            first_level = pathlib.Path(file_path).relative_to(
                base_directory).parts[0]
            if (include_dirs is not None) and (first_level not in include_dirs):
                continue

            # Exclude dirs
            exclude = False
            if exclude_dirs is not None:
                for dir in exclude_dirs:
                    if dir in str(file_path):
                        exclude = True
            if exclude:
                continue

            if os.path.isfile(file_path):
                extension = str(file_path).split(".")[-1]
                if include_extensions is None or extension in include_extensions:
                    archive.write(
                        file_path,
                        arcname=file_path  # .relative_to(base_directory)
                    )
            else:
                archive.write(
                    file_path,
                    arcname=file_path  # .relative_to(base_directory)
                )


def get_extensions_recursive(base_directory: str) -> set[str]:
    pattern = os.path.join("**", "*.*")
    all_files = glob.glob(pattern, root_dir=base_directory, recursive=True)
    return {path.split(".")[-1] for path in all_files}


def worker_initializer(lock):
    tqdm.tqdm.set_lock(lock)

    current = mp.current_process()
    pos = current._identity[0]
    os.environ["TQDM_POSITION"] = str(pos)


def worker_func(args):
    create_zip(**args)


if __name__ == "__main__":
    os.environ["TQDM_POSITION"] = "0"

    parser = argparse.ArgumentParser()

    parser.add_argument("base_directory",
                        help="Directory to export zips",
                        type=str)

    args = parser.parse_args()

    base_directory = args.base_directory

    experiment_name = os.path.basename(base_directory)

    export_directory = experiment_name+"_EXPORT"
    os.makedirs(export_directory, exist_ok=True)

    extensions = get_extensions_recursive(base_directory)

    lock = mp.RLock()
    tqdm.tqdm.set_lock(lock)

    n_process = min(mp.cpu_count(), len(extensions)+1)
    pool = ProcessPoolExecutor(n_process,
                               initializer=worker_initializer,
                               initargs=(lock,))

    futures: list[Future] = []
    for extension in extensions:
        args = {"file_name": experiment_name+"_"+extension,
                "base_directory": base_directory,
                "export_directory": export_directory,
                "include_extensions": [extension],
                "exclude_dirs": ["final_results"]}
        future = pool.submit(worker_func, args)
        futures.append(future)

    if os.path.isdir(os.path.join(base_directory, "final_results")):
        args = {"file_name": experiment_name+"_final_results",
                "base_directory": base_directory,
                "export_directory": export_directory,
                "include_dirs": ["final_results"]}
        future = pool.submit(worker_func, args)
        futures.append(future)

    for future in tqdm.tqdm(as_completed(futures), position=0, desc="Compressing Results"):
        try:
            future.result()
        except Exception as e:
            print("ERROR job exception. Stopping executor.")

            for p in pool._processes.values():
                p.terminate()
            pool.shutdown(wait=False, cancel_futures=True)

            raise e
