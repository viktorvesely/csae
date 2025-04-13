from pathlib import Path
import numpy as np
import tqdm
from datasets import load_dataset

def save_locally_in_batches(output_folder: Path, split: str, N: int, batch_size: int = 1000):
    output_folder.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(
        "lczero-planning/activations",
        "lc0-10-4238.onnx-policy_lc0-10-4238.onnx_9",
        split=split,
        streaming=True,
    )

    ds = ds.select_columns(["root_act", "opt_act", "sub_act"]).with_format("torch")

    root_acts, opt_acts, sub_acts, indices = [], [], [], []
    batch_start = 0

    for i, row in enumerate(tqdm.tqdm(ds, total=N)):
        root_acts.append(np.array(row["root_act"]).flatten().astype(np.float16))
        opt_acts.append(np.array(row["opt_act"]).flatten().astype(np.float16))
        sub_acts.append(np.array(row["sub_act"]).flatten().astype(np.float16))
        indices.append(i)

        if len(root_acts) == batch_size or batch_start + len(root_acts) == N:
            batch_end = batch_start + len(root_acts)
            indices_np = np.array(indices)
            permutation = np.random.permutation(len(root_acts))
            root_acts_np = np.stack(root_acts)[permutation]
            opt_acts_np = np.stack(opt_acts)[permutation]
            sub_acts_np = np.stack(sub_acts)[permutation]
            indices_np = indices_np[permutation]

            np.save(output_folder / f"root_act_rows_{batch_start}_{batch_end}.npy", root_acts_np)
            np.save(output_folder / f"opt_act_rows_{batch_start}_{batch_end}.npy", opt_acts_np)
            np.save(output_folder / f"sub_act_rows_{batch_start}_{batch_end}.npy", sub_acts_np)
            np.save(output_folder / f"indices_rows_{batch_start}_{batch_end}.npy", indices_np)

            root_acts, opt_acts, sub_acts, indices = [], [], [], []
            batch_start = batch_end

        if batch_start >= N:
            break

if __name__ == "__main__":
    split = "train"
    save_locally_in_batches(Path(f"../data/{split}_activations"), split, N=250_000, batch_size=10_000)
