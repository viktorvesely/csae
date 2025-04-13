from pathlib import Path
from typing import Literal
import numpy as np
from cnnsae_simple import CSAE
import torch
from torch.utils.data import DataLoader, Dataset
import tqdm
import tensorboard as ts
from numpy_loader import chunk_loader, chunk_loader_root_only
import argparse

activation_shape = (192, 8, 8)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CNNDataset(Dataset):
    def __init__(self, root, opt, sub):
        self.root = root
        self.opt = opt
        self.sub = sub

    def __len__(self):
        return self.root.shape[0]

    def __getitem__(self, idx):

        r = self.root[idx, :]
        o = self.opt[idx, :]
        s = self.sub[idx, :]

        r = np.reshape(r, (activation_shape[0], activation_shape[1], activation_shape[2]))
        o = np.reshape(o, (activation_shape[0], activation_shape[1], activation_shape[2]))
        s = np.reshape(s, (activation_shape[0], activation_shape[1], activation_shape[2]))

        optimal = np.concatenate((r, o), axis=0)
        suboptimal = np.concatenate((r, s), axis=0)

        optimal = torch.tensor(optimal, dtype=torch.float)
        suboptimal = torch.tensor(suboptimal, dtype=torch.float)

        return optimal, suboptimal

class CNNDatasetRoot(Dataset):
    def __init__(self, root):
        self.root = root

    def __len__(self):
        return self.root.shape[0]

    def __getitem__(self, idx):
        r = self.root[idx, :]
        r = np.reshape(r, (activation_shape[0], activation_shape[1], activation_shape[2]))
        r = torch.tensor(r, dtype=torch.float)

        return r

def validation_loss(model: CSAE, validation_loader: DataLoader, root_only: bool):
    model.eval()

    l_reco = []
    l_contrastive = []
    l_sparsity = []

    with torch.no_grad():
        for i_batch, batch in tqdm.tqdm(enumerate(validation_loader), total=len(validation_loader), disable=True):
            if i_batch > 15:
                break

            if root_only:
                x_batch = batch.to(device)
            else:
                optimal, suboptimal = batch
                x_batch = torch.cat((optimal, suboptimal), dim=0)
                x_batch = x_batch.to(device)

            encoded, decoded = model(x_batch)
            l_reco.append(model.reconstructive_loss(x_batch, decoded).item())
            l_sparsity.append(model.sparsity_loss(encoded).item())
            l_contrastive.append(0 if root_only else model.contrastive_loss(encoded).item())

    return np.mean(l_reco), np.mean(l_sparsity), np.mean(l_contrastive)

def training(
    top_r: int = 10,
    ls_factor: int = 5,
    small: bool=False,
    name: str ="experiment",
    GFCdiv: int = 1,
    checkpoint: str | None = None,
    root_only: bool = True,
    sparsity_lambda: float = 5.0,
    file_names: Literal["theirs", "ours"] = "ours"
    ):


    # We have a system in place which detects if a model's loss epxloded
    # If so we restore previous chcekpoint
    loss_restore_model_threshold = 0.05
    last_reco_loss = float("inf")

    in_channels = activation_shape[0] * (1 if root_only else 2)

    model = CSAE(
        in_channels=in_channels,
        ls_factor=ls_factor,
        r_values=(top_r, ),
        sparsity_lambda=sparsity_lambda,
        contrastive_lambda=0.0,
        GFC_divisor=GFCdiv
    )

    if checkpoint is None:
        train_dir = ts.make_train_dir(name=name, path=Path(__file__).parent / "experiments")
        ts.copy_script_to_train_folder(train_dir, script_name="cnnsae_simple.py", script_name2="csae_train.py")

    else:
        train_dir = Path(__file__).parent / "experiments" / checkpoint
        cp = torch.load(train_dir / "checkpoint.pt", weights_only=True)
        model.load_state_dict(cp["model_state_dict"])


    # Small = local not small = Habrok
    if small:
        batch_size = 200
    else:
        batch_size = 500

    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(
      model.parameters(),
      lr=5 * 1e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        threshold_mode="abs",
        patience=3,
        threshold=0.01,
        factor=0.5
    )

    CustomDataset: Dataset = CNNDatasetRoot if root_only else CNNDataset
    chunk_generator = chunk_loader_root_only if root_only else chunk_loader
    data_path = Path(__file__).parent / "data"
    activations_folder = (data_path / "our_activations") if file_names == "ours" else (data_path / "train_activations")

    MAX_EPOCHS = 1_000
    for epoch in range(MAX_EPOCHS):
        print(f"epoch: {epoch}")

        l_reco = []
        l_contrastive = []
        l_sparsity = []
    
        chunk_iterator = chunk_generator(activations_folder, file_names=file_names)
        validation_chunk = next(chunk_iterator)
        validation_dataset = CustomDataset(*validation_chunk)
        validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)


        for i_chunk, chunk in enumerate(chunk_iterator):
            train_dataset = CustomDataset(*chunk)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            for batch in tqdm.tqdm(train_loader, total=len(train_loader), disable=not small):

                if root_only:
                    x_batch = batch.to(device)
                else:
                    optimal, suboptimal = batch
                    x_batch = torch.cat((optimal, suboptimal), dim=0)
                    x_batch = x_batch.to(device)

                encoded, decoded = model(x_batch)
                reconstruction_loss = model.reconstructive_loss(x_batch, decoded)
                sparsity_loss = model.sparsity_loss(encoded)

                if root_only:
                    contrastive_loss = 0
                else:
                    contrastive_loss = model.contrastive_loss(encoded)

                loss = reconstruction_loss + contrastive_loss + sparsity_loss

                optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                l_reco.append(reconstruction_loss.item())
                l_contrastive.append(sparsity_loss.item())
                l_sparsity.append(0 if root_only else contrastive_loss.item())

            train_reco, train_sparsity, train_contrastive = np.mean(l_reco), np.mean(l_sparsity), np.mean(l_contrastive)
            print(f"{train_reco:.3f}, {train_sparsity:.3f}, {train_contrastive:.3f}")
            l_reco = []
            l_contrastive = []
            l_sparsity = []

            delta_loss = train_reco - last_reco_loss

            if delta_loss > loss_restore_model_threshold:
                print(f"[Chunk_{i_chunk}] increased loss too much, restoring previous checkpoint")
                ts.load_checkpoint(model, checkpoint_path=train_dir / "checkpoint.pt")
            else:
                ts.save_checkpoint(model, optimizer=None, dictionary_size=None, checkpoint_path=train_dir / "checkpoint.pt")
                last_reco_loss = min(train_reco, last_reco_loss)

            with open(train_dir / f"reco_loss.txt", "a") as f:
                f.write(f"{train_reco:.3f},{train_sparsity:.3f},{train_contrastive:.3f}\n")


        valid_reco, valid_sparsity, valid_contrastive  = validation_loss(model, validation_loader, root_only)
        model.train()
        print(f"{valid_reco:.3f}, {valid_sparsity:.3f}, {valid_contrastive:.3f}")

        scheduler.step(valid_reco)

        print(f"[Validation] {valid_reco:.3f},{valid_sparsity:.3f},{valid_contrastive:.3f}")
        with open(train_dir / f"reco_loss.txt", "a") as f:
            f.write(f"Epoch {epoch} finished\n")
            f.write(f"[Validation] {valid_reco:.3f},{valid_sparsity:.3f},{valid_contrastive:.3f}\n")

        if valid_reco.item() < 0.01:
            break



if __name__ == "__main__":

    # Most of the arguments do not do anything and are here for backwards compatibility
    parser = argparse.ArgumentParser()
    parser.add_argument("--lsfactor", type=int, required=False, default=5)
    parser.add_argument("--topr", type=int, required=False, default=10)
    parser.add_argument("--name", type=str, required=False, default="experiment")
    parser.add_argument("--checkpoint", type=str, required=False, default="")
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--GFCdiv", type=int, required=False, default=1)
    parser.add_argument("--sparsity", type=float, required=False, default=5)

    args = parser.parse_args()
    checkpoint = args.checkpoint
    if checkpoint == "":
        checkpoint = None

    training(
        top_r=args.topr,
        ls_factor=args.lsfactor,
        small=args.small,
        name=args.name,
        GFCdiv=args.GFCdiv,
        checkpoint=checkpoint,
        sparsity_lambda=args.sparsity
    )
