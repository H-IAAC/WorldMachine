import os
import shutil

from .utils import file_map, page_source, project_root, remove_if_exists


def pre_build():

    for origin_path, dest_path in file_map.items():
        remove_if_exists(dest_path)
        shutil.copyfile(origin_path, dest_path)

    shutil.copytree(os.path.join(project_root, "examples"),
                    os.path.join(page_source, "examples", "notebooks"))

    shutil.copytree(os.path.join(project_root, "experiments", "reports"),
                    os.path.join(page_source, "reports", "reports"))
