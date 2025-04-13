import re
from pathlib import Path
from typing import Literal
import numpy as np


def get_sorted_chunks(activations_folder: Path, file_names: Literal["theirs", "ours"] = "theirs") -> dict[str, list[Path]]:

    if file_names == "theirs":
        regex = r"(opt|sub|root)_act_rows_(\d+)_\d+"
        names = ["root", "opt", "sub"]
    else:
        regex = r"(root|optimal|suboptimal)_(\d+)"
        names = ["root", "optimal", "suboptimal"]

    files = {key: [] for key in names}

    for f in activations_folder.iterdir():

        res = re.findall(regex , f.name)
        if len(res) == 0:
            continue

        name, start = res[0]
        files[name].append((int(start), f))

    sorted_files = {key: [] for key in names}
    for name in names:
        sorted_info = sorted(files[name], key=lambda x: x[0])
        sorted_files[name] = [p for _, p in sorted_info]

    return sorted_files, names

def chunk_loader(activations_folder: Path, file_names: Literal["theirs", "ours"] = "theirs"):

    sorted_files, names = get_sorted_chunks(activations_folder, file_names)

    data = (sorted_files[names[0]], sorted_files[names[1]], sorted_files[names[2]])

    for rootp, optp, subp in zip(*data, strict=True):
        root = np.load(rootp)
        opt = np.load(optp)
        sub = np.load(subp)

        yield root, opt, sub


def chunk_loader_root_only(activations_folder: Path, file_names: Literal["theirs", "ours"] = "theirs"):

    sorted_files, names = get_sorted_chunks(activations_folder, file_names)

    roots = sorted_files[names[0]]

    for rootp in roots:
        root = np.load(rootp)
        yield (root,)
