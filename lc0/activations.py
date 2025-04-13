"""
Slightly inspired by https://github.com/Xmaster6y/lczero-planning/blob/main/scripts/datasets/make_trajectories_dataset.py
"""

from pathlib import Path
from typing import Callable, Iterator
import torch
import numpy as np
from lczerolens import LczeroModel, LczeroBoard
import random
import io
import tqdm
import zstandard
import chess
import chess.pgn
import re

from torch.utils.data import TensorDataset, DataLoader

ACT_LAYER = ".block14/conv2/mish"
BATCH_SIZE = 3_000

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_lc0():
    path = str(Path(__file__).parent / "t79.onnx")
    model = LczeroModel.from_path(path).to(device)
    return model

def extract_activations(model: LczeroModel, inputs: torch.Tensor, batch_size: int = BATCH_SIZE):

    dataset = TensorDataset(inputs)
    loader = DataLoader(dataset, batch_size)
    list_activations = []

    for (x,) in loader:
        with torch.no_grad(), model.trace() as tracer:
            x = x.to(device)
            with tracer.invoke(x):
                act_module = None
                for name, module in model.named_modules():
                    if name == ACT_LAYER:
                        act_module = module
                        break
                if act_module is None:
                    raise ValueError("Activation layer not found in model.")

                activations = act_module.output.save()

        list_activations.append(activations.cpu())

    return torch.cat(list_activations, dim=0)

def simulate_trajectory(
    model: LczeroModel,
    boards: list[LczeroBoard],
    sampler: Callable,
    depth: int,
    batch_size: int = BATCH_SIZE
):
    boards_sim = [board.copy() for board in boards]

    for _ in range(depth):
        indices = torch.arange(len(boards_sim))
        dataset = TensorDataset(indices)
        loader = DataLoader(dataset, batch_size=batch_size)

        for (batch_indices,) in loader:

            # Progress only those without gameover
            batch_boards = []
            valid_indices = []
            for i in batch_indices:
                if not boards_sim[i].is_game_over():
                    batch_boards.append(boards_sim[i])
                    valid_indices.append(i)

            if not batch_boards:
                continue

            batch_inputs = torch.stack([b.to_input_tensor() for b in batch_boards]).to(device)
            with torch.no_grad():
                outputs = model(batch_inputs)

            batch_moves = sampler(outputs, batch_boards)


            for idx, move in zip(valid_indices, batch_moves):
                if move is not None and not boards_sim[idx].is_game_over():
                    boards_sim[idx].push(move)
    return boards_sim


def simulate_trajectory_save_moves(
    model: LczeroModel,
    boards: list[LczeroBoard],
    sampler: Callable,
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

            # Progress only those without gameover
            batch_boards = []
            valid_indices = []
            for i in batch_indices:
                if not boards_sim[i].is_game_over():
                    batch_boards.append(boards_sim[i])
                    valid_indices.append(i)

            if not batch_boards:
                continue

            batch_inputs = torch.stack([b.to_input_tensor() for b in batch_boards]).to(device)
            with torch.no_grad():
                outputs = model(batch_inputs)

            batch_moves = sampler(outputs, batch_boards)

            for idx, move in zip(valid_indices, batch_moves):
                if move is not None and not boards_sim[idx].is_game_over():
                    boards_sim[idx].push(move)

                    move = boards_sim[idx].peek()

                    uci_str = move.uci()
                    if move.promotion:
                        uci_str += chess.piece_symbol(move.promotion)

                    trajectories[idx].append(uci_str)

    return trajectories



def mask_legal_moves(policy_logits: torch.Tensor, board: LczeroBoard):
    # TODO this is stupid, fill the -inf straight from indicies
    legal_moves: torch.Tensor = board.get_legal_indices().to(torch.int64)
    n_legal = legal_moves.numel()
    legal_mask = torch.zeros_like(policy_logits, dtype=torch.bool)
    legal_mask[legal_moves] = True
    legal_logits = policy_logits.masked_fill(~legal_mask, float('-inf'))
    return legal_logits, n_legal

def optimal_sampler(outputs, boards: list[LczeroBoard]):
    # Select best
    policies = outputs["policy"]
    chosen: list[chess.Move | None] = []

    for i, board in enumerate(boards):
        legal_logits, n_legal = mask_legal_moves(policies[i], board)

        if n_legal == 0:
            chosen.append(None)
            continue

        idx = torch.argmax(legal_logits).item()
        move = board.decode_move(idx)
        chosen.append(move)

    return chosen

def suboptimal_sampler(outputs, boards: list[LczeroBoard]):
    # Sets best o -inf and sample the rest
    policies = outputs["policy"]
    chosen = []
    for i, board in enumerate(boards):

        legal_logits, n_legal = mask_legal_moves(policies[i], board)

        idx = None
        if n_legal == 0:
            chosen.append(None)
            continue
        elif n_legal == 1:
            idx = torch.argmax(legal_logits)
        else:
            best_idx = torch.argmax(legal_logits)
            legal_logits[best_idx] = -float("inf")
            legal_probs = torch.softmax(legal_logits, dim=-1)
            idx = torch.multinomial(legal_probs, num_samples=1).item()

        move = board.decode_move(idx)
        chosen.append(move)
    return chosen

def process_fen_chunk(model: LczeroModel, chunk_fens: list[str], chunk_name: str, folder: Path, depth: int, root_only: bool = False):

    boards_root = []
    for fen in chunk_fens:
        board = LczeroBoard(fen)
        boards_root.append(board)

    root_inputs = torch.stack([board.to_input_tensor() for board in boards_root])
    root_acts = extract_activations(model, root_inputs)

    root_np = root_acts.cpu().to(torch.float16).numpy()
    root_path = folder / f"root_{chunk_name}.npy"
    np.save(root_path, root_np)

    if root_only:
        return

    boards_opt = simulate_trajectory(model, boards_root, optimal_sampler, depth)
    boards_sub = simulate_trajectory(model, boards_root, suboptimal_sampler, depth)

    opt_inputs = torch.stack([b.to_input_tensor() for b in boards_opt])
    opt_acts = extract_activations(model, opt_inputs)
    opt_np = opt_acts.to(torch.float16).numpy()
    opt_path = folder / f"optimal_{chunk_name}.npy"
    np.save(opt_path, opt_np)

    sub_inputs = torch.stack([b.to_input_tensor() for b in boards_sub])
    sub_acts = extract_activations(model, sub_inputs)
    sub_np = sub_acts.cpu().to(torch.float16).numpy()
    sub_path = folder / f"suboptimal_{chunk_name}.npy"
    np.save(sub_path, sub_np)


def save_activations(fens_name: str = "fens", out_name: str = "our_activations"):

    fens_folder = Path(__file__).parent.parent / "data" / fens_name
    out_folder = Path(__file__).parent.parent / "data" / out_name

    model = load_lc0()

    for ent in tqdm.tqdm(list(fens_folder.iterdir())):

        if (not ent.is_file()) or (not ent.name.endswith(".npy")):
            continue

        res = re.match(r"fens_([0-9]+)\.npy", ent.name)
        i = int(res.group(1))

        fens = np.load(ent)
        process_fen_chunk(model, fens, str(i), out_folder, depth=14, root_only=True)



def process_trajectories(model: LczeroModel, chunk_fens: list[str], chunk_name: str, folder: Path, depth: int):

    boards_root = []
    for fen in chunk_fens:
        board = LczeroBoard(fen)
        boards_root.append(board)

    optimal_trajectory = simulate_trajectory_save_moves(model, boards_root, optimal_sampler, depth)

    for trajectory in optimal_trajectory:
        l = len(trajectory)
        if l < (depth + 1):
            add = (depth + 1) - l
            trajectory.extend([""] * add)

    optimal_trajectory = np.array(optimal_trajectory)

    np.save(folder / f"trajectory_{chunk_name}", optimal_trajectory)


def save_trajectories(fens_name: str = "fens_test", out_name: str = "test_trajectories"):

    fens_folder = Path(__file__).parent.parent / "data" / fens_name
    out_folder = Path(__file__).parent.parent / "data" / out_name

    model = load_lc0()

    for ent in tqdm.tqdm(list(fens_folder.iterdir())):

        if (not ent.is_file()) or (not ent.name.endswith(".npy")):
            continue

        res = re.match(r"fens_([0-9]+)\.npy", ent.name)
        i = int(res.group(1))

        fens = np.load(ent)
        process_trajectories(model, fens, str(i), out_folder, depth=8)


def generate_fens_batches(file_path: Path, batch_size: int, max_games: int) -> Iterator[str]:
    fens: list[str] = []


    with file_path.open("rb") as f:
        dctx = zstandard.ZstdDecompressor()

        with dctx.stream_reader(f) as reader:
            stream = io.TextIOWrapper(reader, encoding="utf-8")
            games_processed = 0
            while games_processed < max_games:
                game = chess.pgn.read_game(stream)

                if game is None:
                    break

                try:
                    white_elo = int(game.headers.get("WhiteElo", "0"))
                    black_elo = int(game.headers.get("BlackElo", "0"))
                except ValueError:
                    continue

                if (white_elo + black_elo) / 2 <= 1400:
                    continue

                moves = list(game.mainline_moves())
                full_moves = (len(moves) + 1) // 2
                if full_moves <= 5 or full_moves > 100:
                    continue

                board = game.board()
                for move in moves:
                    board.push(move)
                    if random.random() > (1/2):
                        continue

                    fens.append(board.fen())
                    if len(fens) >= batch_size:
                        break

                games_processed += 1
                if len(fens) >= batch_size:
                    random.shuffle(fens)
                    yield fens
                    fens = []


def save_fens(n_batches: int, batch_size: int = 10_000, skip_batches: int = 0, fens_folder: str = "fens"):
    out_folder = Path(__file__).parent.parent / "data" / fens_folder
    database = Path(__file__).parent.parent  / "viktor" / "lichess_db_standard_rated_2024-08.pgn.zst"

    iterator = generate_fens_batches(database, batch_size=batch_size, max_games=20_000_000)
    for i in tqdm.tqdm(list(range(n_batches + skip_batches))):

        if i < skip_batches:
            continue

        batch = np.array(next(iterator))
        np.save(out_folder / f"fens_{i}.npy", batch)



if __name__ == "__main__":

    # bs = 10_000
    # train_all_fens = 250_000
    # train_batches = train_all_fens // bs

    # test_fens = 50_000
    # test_batches = test_fens // bs
    # save_fens(test_batches, bs, skip_batches=train_batches, fens_folder="fens_test")

    # save_activations(fens_name="fens_test", out_name="our_test")

    save_trajectories()

