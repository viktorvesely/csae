import argparse
import re
import numpy as np
from cnnsae_simple import CSAE

import torch
from torch.utils.data import DataLoader
import tqdm
from pathlib import Path
import pandas as pd

from csae_train import CNNDatasetRoot
from numpy_loader import chunk_loader_root_only


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def traj_loader(fens_folder: Path):

    files = []

    for ent in fens_folder.iterdir():

        if (not ent.is_file()) or (not ent.name.endswith(".npy")):
            continue

        res = re.match(r"trajectory_([0-9]+)\.npy", ent.name)
        i = int(res.group(1))

        files.append((i, ent))

    files = sorted(files, key=lambda x: x[0])

    for i_chunk, ent in files:
        yield i_chunk, np.load(ent)


def load_sae(model_name: str = "CNN_SAE_factor10"):

    folder = Path(__file__).parent / "experiments" / model_name

    model = CSAE(in_channels=192, ls_factor=10)
    cp = torch.load(folder / "checkpoint.pt", weights_only=True)
    model.load_state_dict(cp["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model



def test_inference(model_name: str = "CNN_SAE_factor10"):

    data_folder = Path(__file__).parent  / "data"
    test_activations = data_folder  / "our_test"
    trajectory_folder = data_folder / "test_trajectories"
    latent_folder = data_folder / "test_latents"

    model = load_sae(model_name)


    l_reco = []
    l_sparsity = []
    T = 0.05
    sparsities = []
    n_maus = 10


    df = {"fens": [], "move": []}

    for imau in range(n_maus):
        df[f"mau{imau + 1}"] = []
        df[f"a{imau + 1}"] = []

    with torch.no_grad():
        for chunk, (i_chunk, trajectory) in zip(chunk_loader_root_only(test_activations, file_names="ours"), traj_loader(trajectory_folder), strict=True):

            dataset = CNNDatasetRoot(*chunk)
            bs = 300
            loader = DataLoader(dataset, batch_size=bs)

            for i_batch, batch in tqdm.tqdm(enumerate(loader), total=len(loader)):

                x_batch = batch.to(device)

                encoded, decoded = model(x_batch)
                l_reco.append(model.reconstructive_loss(x_batch, decoded).item())
                l_sparsity.append(model.sparsity_loss(encoded).item())

                n_bellow = (encoded > T).sum()
                sparsity = n_bellow / x_batch.numel()
                sparsities.append(sparsity.item())

                # (n_batch, channels, w, h)
                latents = encoded.cpu().numpy().astype(np.float16)

                latents = latents.astype(np.float16)
                flattened = latents.reshape((latents.shape[0], -1))

                # Finds top n_maus active unit
                # Argpartiotion is much much faster than sort
                part_indices = np.argpartition(-flattened, n_maus, axis=1)[:, :n_maus]
                row_inds = np.arange(flattened.shape[0])[:, np.newaxis]

                sorted_order = np.argsort(-flattened[row_inds, part_indices], axis=1)

                mau_inds = part_indices[row_inds, sorted_order]
                maus_vals = flattened[row_inds, mau_inds]

                for imau in range(n_maus):
                    df[f"mau{imau + 1}"].extend(mau_inds[:, imau])
                    df[f"a{imau + 1}"].extend(maus_vals[:, imau])

                np.save(latent_folder / f"latent_{i_chunk}_{i_batch}.npy", latents)

            df["fens"].extend(trajectory[:, 0])
            df["move"].extend(trajectory[:, 1])



    df = pd.DataFrame(df)
    df.to_parquet(data_folder / "mau.parquet", index=False)
    mean_reco, mean_l_sparsity, mean_sparsity = np.mean(l_reco), np.mean(l_sparsity), np.mean(sparsities)

    print(f"reco={mean_reco}, sparsity={mean_sparsity}")



if __name__  == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=False, default="CNN_SAE_factor10")
    args = parser.parse_args()

    test_inference(args.checkpoint)