import os

project_root = os.path.join("..", "..")
page_source = os.path.join("..", "source")

file_map = {
    os.path.join(project_root, "README.md"): os.path.join(page_source, "README.md")}


def remove_if_exists(path: str) -> None:
    try:
        os.remove(path)
    except:
        pass
