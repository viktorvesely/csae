import re
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path

import tqdm

from cnnsae_simple import CSAE
from lczerolens import LczeroModel, LczeroBoard
from activations import load_lc0, mask_legal_moves, extract_activations

BATCH_SIZE = 300
ACT_LAYER = ".block14/conv2/mish"
OUTPUT_LAYER = ".output/policy"


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_sae():

    model_name = "CNN_SAE_factor10"
    folder = Path(__file__).parent.parent / "experiments" / model_name

    model = CSAE(in_channels=192, ls_factor=10)
    cp = torch.load(folder / "checkpoint.pt", weights_only=True)
    model.load_state_dict(cp["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model

def mind_control(model: LczeroModel, inputs: torch.Tensor, H: torch.Tensor, batch_size: int = BATCH_SIZE) -> torch.Tensor:

    dataset = TensorDataset(inputs, H)
    loader = DataLoader(dataset, batch_size)
    list_outputs = []

    for x, h in loader:
        with torch.no_grad(), model.trace() as tracer:
            x = x.to(device)
            h = h.to(device)

            with tracer.invoke(x):
                act_module = None
                out_module = None
                for name, module in model.named_modules():
                    if name == ACT_LAYER:
                        act_module = module

                    if name == OUTPUT_LAYER:
                        out_module = module

                if act_module is None or out_module is None:
                    raise ValueError("Activation or output layer not found in model.")

                act_module.output[:] = h

                outputs = out_module.output.save()

        list_outputs.append(outputs.cpu())

    return torch.cat(list_outputs, dim=0)


def encode(sae: CSAE, H: torch.Tensor, batch_size: int = BATCH_SIZE) -> torch.Tensor:

    dataset = TensorDataset(H)
    loader = DataLoader(dataset, batch_size)
    list_encode = []

    with torch.no_grad():

        for (h,) in loader:
            h = h.to(device)

            e = sae.encoder(h)
            list_encode.append(e.cpu())

    return torch.cat(list_encode, 0)

def decode(sae: CSAE, latents: torch.Tensor, batch_size: int = BATCH_SIZE) -> torch.Tensor:

    dataset = TensorDataset(latents)
    loader = DataLoader(dataset, batch_size)
    list_h = []

    with torch.no_grad():

        for (latent,) in loader:
            latent = latent.to(device)

            d = sae.decoder(latent)
            list_h.append(d.cpu())

    return torch.cat(list_h, 0)

def latent_manipulation(latents: torch.Tensor, indices: torch.Tensor, factors: torch.Tensor):

    assert latents.is_contiguous()

    bs = latents.shape[0]
    latents = latents.view(bs, -1)

    # Set the latent manipulation to all batches at once
    # The code uses advanced indexing to avoid for loop
    # That is why it is so ugly
    batch_indices = torch.arange(bs, device=latents.device).unsqueeze(1).expand(bs, indices.size(0))
    latent_indices = indices.unsqueeze(0).expand(bs, indices.size(0))

    latents[batch_indices, latent_indices] = factors


def get_latent_manipulator(*inds: int, factor: float):
    inds = torch.tensor(inds)
    return inds, torch.full_like(inds, factor, dtype=torch.float)



def optimal_sampler(policies, boards):
    chosen = []

    for i, board in enumerate(boards):
        legal_logits, n_legal = mask_legal_moves(policies[i], board)

        if n_legal == 0:
            chosen.append(None)
            continue

        idx = torch.argmax(legal_logits).item()
        move = board.decode_move(idx)
        chosen.append(move)
    return chosen

def simulate_altered_trajectory_save_moves(
    lc0: LczeroModel,
    sae: CSAE,
    boards: list[LczeroBoard],
    lm_indices: torch.Tensor,
    lm_factors: torch.Tensor,
    depth: int,
    batch_size: int = BATCH_SIZE
) -> list[list[str]]:

    boards_sim = [board.copy() for board in boards]
    trajectories = [[board.fen()] for board in boards]

    for _ in range(depth):
        indices = torch.arange(len(boards_sim))
        dataset = TensorDataset(indices)
        loader = DataLoader(dataset, batch_size=batch_size)

        for (batch_indices,) in loader:
            batch_indices = batch_indices.tolist()

            # Progress only boards wich did not end
            batch_boards = []
            valid_indices = []
            for i in batch_indices:
                if not boards_sim[i].is_game_over():
                    batch_boards.append(boards_sim[i])
                    valid_indices.append(i)

            if not batch_boards:
                continue

            batch_inputs = torch.stack([b.to_input_tensor() for b in batch_boards])

            # Where the magic happens
            H = extract_activations(lc0, batch_inputs, batch_size=BATCH_SIZE)
            latents = encode(sae, H)
            latent_manipulation(latents, lm_indices, lm_factors)
            H_altered = decode(sae, latents)

            print((H - H_altered).abs().mean())

            policies = mind_control(lc0, batch_inputs, H_altered)

            batch_moves = optimal_sampler(policies, batch_boards)

            for idx, move in zip(valid_indices, batch_moves):
                if move is not None and not boards_sim[idx].is_game_over():
                    boards_sim[idx].push(move)
                    trajectories[idx].append(move.uci())

    return trajectories

def altered_trajectories(
    folder: Path,
    sae: CSAE,
    lc0: LczeroModel,
    chunk_fens: list[str],
    chunk_name: str,
    lm_indices: torch.Tensor,
    lm_factors: torch.Tensor,
    depth: int=8
):

    boards_root = []
    for fen in chunk_fens:
        board = LczeroBoard(fen)
        boards_root.append(board)

    altered_trajectory = simulate_altered_trajectory_save_moves(lc0, sae, boards_root, lm_indices, lm_factors, depth)

    # Make it blocky (numpy does not allow heterogenous arrays)
    for trajectory in altered_trajectory:
        l = len(trajectory)
        if l < (depth + 1):
            add = (depth + 1) - l
            trajectory.extend([""] * add)

    altered_trajectory = np.array(altered_trajectory)

    np.save(folder / f"trajectory_{chunk_name}", altered_trajectory)


def save_trajectories(fens_name: str = "fens_test", out_name: str = "altered_trajectories"):

    fens_folder = Path(__file__).parent.parent / "data" / fens_name
    out_folder = Path(__file__).parent.parent / "data" / out_name

    out_folder.mkdir(parents=True, exist_ok=True)

    lc0 = load_lc0()
    sae = load_sae()

    lm_indices, lm_factors = get_latent_manipulator(105472, 57536, 63744, 61049, factor=2)

    for ent in tqdm.tqdm(list(fens_folder.iterdir())):

        if (not ent.is_file()) or (not ent.name.endswith(".npy")):
            continue

        res = re.match(r"fens_([0-9]+)\.npy", ent.name)
        i = int(res.group(1))

        fens = np.load(ent)

        # Used for debugging
        fens = fens[:1_000]

        with torch.no_grad():
            altered_trajectories(out_folder, sae, lc0, fens, str(i), lm_indices, lm_factors)

        # Used for debugging
        assert False, "Remove me in the future"

if __name__ == "__main__":
    save_trajectories()