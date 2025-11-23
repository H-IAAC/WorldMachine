import os
import shutil

from .utils import file_map, page_source, project_root, remove_if_exists


def post_build(*args, **kargs):
    original_cwd = os.getcwd()
    os.chdir(os.path.join(original_cwd, "source"))

    for dest_path in file_map.values():
        remove_if_exists(dest_path)

    shutil.rmtree(os.path.join(page_source, "examples", "notebooks"))
    shutil.rmtree(os.path.join(page_source, "reports", "reports"))

    os.chdir(original_cwd)
