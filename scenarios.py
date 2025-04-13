import pandas as pd
from datasets import load_dataset
from pathlib import Path
import numpy as np

import chess
from typing import NamedTuple, List

import tqdm


class Scenario(NamedTuple):
    scenario_name: str
    move_index: int

class ScenarioRecord(NamedTuple):
    scenario_name: str
    move_index: int
    row_id: int


def pawn_promotion(move: chess.Move) -> bool:
    raise NotImplementedError("Implementation was lost unfortunately")


def queen_loss(i: int, captured: dict, valid_indices: List[int], moves: List[str]) -> bool:
    if captured.get(i) == chess.QUEEN:
        pos = valid_indices.index(i)
        if pos + 1 < len(valid_indices):
            last_move = chess.Move.from_uci(moves[i - 1]) if i - 1 in valid_indices else None
            if (
                captured.get(i - 1) != chess.QUEEN
                and captured.get(i + 1) != chess.QUEEN
                and not (last_move and last_move.promotion == chess.QUEEN)
            ):
                return True
    return False

def knight_fork(board: chess.Board, move: chess.Move) -> bool:

    piece = board.piece_at(move.to_square)
    if not piece or piece.piece_type != chess.KNIGHT:
        return False
    attacked = board.attacks(move.to_square)
    majors = sum(1 for s in attacked if (board.piece_at(s) and (board.piece_at(s).piece_type in [chess.KING, chess.QUEEN, chess.ROOK]) and (board.piece_at(s).color == board.turn)))
    if majors < 2:
        return False
    protected = any(True for _ in board.attackers(not board.turn, move.to_square))
    if not protected:
        return False
    lesser = sum(board.piece_at(s) and board.piece_at(s).piece_type in [chess.PAWN, chess.BISHOP, chess.KNIGHT] for s in board.attackers(board.turn, move.to_square))


    return lesser == 0

def pawn_double_move(board: chess.Board, move: chess.Move) -> bool:
    piece = board.piece_at(move.from_square)
    
    if not piece or piece.piece_type != chess.PAWN or piece.color != board.turn:
        return False
    
    start_rank = chess.square_rank(move.from_square)
    return abs(start_rank - chess.square_rank(move.to_square)) == 2 and (
        (piece.color == chess.WHITE and start_rank == 1) or
        (piece.color == chess.BLACK and start_rank == 6)
    )


def is_castling(board: chess.Board, move: chess.Move) -> bool:

    if not board.is_castling(move):
        return False
    
    king_square = board.king(board.turn)
    if king_square is None:
        return False  

    castling_data = {
        chess.G1: (chess.H1, [chess.F1, chess.G1]),
        chess.C1: (chess.A1, [chess.D1, chess.C1, chess.B1]),
        chess.G8: (chess.H8, [chess.F8, chess.G8]),
        chess.C8: (chess.A8, [chess.D8, chess.C8, chess.B8])
    }
    
    if move.to_square not in castling_data:
        return False
    
    rook_square, path_squares = castling_data[move.to_square]
    
    rook_piece = board.piece_at(rook_square)
    if not rook_piece or rook_piece.piece_type != chess.ROOK or rook_piece.color != board.turn:
        return False  
    
    if any(board.piece_at(sq) for sq in path_squares) or any(board.is_attacked_by(not board.turn, sq) for sq in path_squares):
        return False

    temp_board = board.copy()
    temp_board.push(move)
    if temp_board.is_check():
        return False

    return True


def discovered_attack(board: chess.Board, move: chess.Move) -> bool:
    print(f"\nChecking move: {move.uci()} for discovered attack...")

    temp_board = board.copy()

    temp_board.push(move)

    opponent_king_square = temp_board.king(not board.turn)
    if opponent_king_square is None:
        return False  

    for sq in chess.SquareSet(temp_board.occupied_co[board.turn]): 
        piece = temp_board.piece_at(sq)
        if piece and piece.piece_type in [chess.ROOK, chess.QUEEN, chess.BISHOP]:
            if is_clear_attack_line(temp_board, sq, opponent_king_square):
                return True
    return False


def is_clear_attack_line(board: chess.Board, start_square: int, target_square: int) -> bool:
    piece = board.piece_at(start_square)
    if not piece:
        return False

    direction = None
    if piece.piece_type == chess.ROOK or piece.piece_type == chess.QUEEN:
        if chess.square_file(start_square) == chess.square_file(target_square):  # Vertical
            direction = 'vertical'
        elif chess.square_rank(start_square) == chess.square_rank(target_square):  # Horizontal
            direction = 'horizontal'
    elif piece.piece_type == chess.BISHOP or piece.piece_type == chess.QUEEN:
        if abs(chess.square_rank(start_square) - chess.square_rank(target_square)) == abs(chess.square_file(start_square) - chess.square_file(target_square)):
            direction = 'diagonal'

    if direction is None:
        return False

    path = get_attack_path(start_square, target_square, direction)

    for sq in path:
        if board.piece_at(sq):
            return False  
    return True


def get_attack_path(start_square: int, target_square: int, direction: str) -> List[int]:
    path = []
    rank_diff = chess.square_rank(target_square) - chess.square_rank(start_square)
    file_diff = chess.square_file(target_square) - chess.square_file(start_square)

    if direction == 'horizontal':  
        step = 1 if file_diff > 0 else -1
        for file in range(chess.square_file(start_square) + step, chess.square_file(target_square), step):
            path.append(chess.square(chess.square_rank(start_square), file))
    
    elif direction == 'vertical':  
        step = 1 if rank_diff > 0 else -1
        for rank in range(chess.square_rank(start_square) + step, chess.square_rank(target_square), step):
            path.append(chess.square(rank, chess.square_file(start_square)))

    elif direction == 'diagonal':  
        step_rank = 1 if rank_diff > 0 else -1
        step_file = 1 if file_diff > 0 else -1
        rank = chess.square_rank(start_square) + step_rank
        file = chess.square_file(start_square) + step_file
        while rank != chess.square_rank(target_square) and file != chess.square_file(target_square):
            path.append(chess.square(rank, file))
            rank += step_rank
            file += step_file
    
    return path

def detect_scenarios(fen: str, moves: List[str]) -> List[Scenario]:
    scenarios = []
    board = chess.Board(fen)
    captured = {}
    valid_indices = []

    for i, uci in enumerate(moves[7:], start=7):
        m = chess.Move.from_uci(uci)
        if board.is_legal(m):
            c = board.piece_at(m.to_square)
            captured[i] = c.piece_type if c else None
            board.push(m)
            valid_indices.append(i)

    board = chess.Board(fen)
    for i in valid_indices:
        m = chess.Move.from_uci(moves[i])
        board.push(m)


        if pawn_promotion(m):
            scenarios.append(Scenario("pawn_promotion", i))

        if queen_loss(i, captured, valid_indices, moves):
            scenarios.append(Scenario("queen_loss", i))

        if knight_fork(board, m):
            scenarios.append(Scenario("knight_fork", i))

        if pawn_double_move(board, m):
            scenarios.append(Scenario("pawn_double_move", i))
        
        if is_castling(board, m):
            scenarios.append(Scenario("castling", i))

    return scenarios


def parse_games(p: Path, split: str):

    df = pd.read_parquet(p)

    records: list[ScenarioRecord] = []

    rollouts = df.loc[~df.duplicated(["root_fen", "moves_opt"], keep="last"), ["root_fen", "moves_opt"]]
    n_rollouts = rollouts.shape[0]

    for _, row in tqdm.tqdm(rollouts.iterrows(), total=n_rollouts):
        fen = row["root_fen"]
        moves = row["moves_opt"]
        moves_list = moves.split(" ")

        df_rollout = df[(df["root_fen"] == fen) & (df["moves_opt"] == moves)]
        scenarios = detect_scenarios(fen, moves_list)

        for scenario in scenarios:
            name, at_move = scenario

            distance = (df_rollout["current_depth"] - at_move).abs()
            i_closest = distance.argmin()
            closest_idx = df_rollout.index[i_closest]
            row_id = df_rollout.loc[closest_idx, "row_id"]

            records.append(ScenarioRecord(name, at_move, row_id))

    records_df = pd.DataFrame(records, columns=["scenario_name", "move_index", "row_id"])
    print(records_df["scenario_name"].value_counts())
    records_df.to_parquet(p.parent / f"{split}_records.parquet", index=False)

def save_locally(p: Path, split: str, N: int):
     # Load the dataset with streaming
    ds = load_dataset(
        "lczero-planning/activations",
        "lc0-10-4238.onnx-policy_lc0-10-4238.onnx_9",
        split=split,
        streaming=True,
    )

    # Select specific columns
    # ds = ds.select_columns(["gameid", "moves", "fen", "root_fen", "opt_fen", "sub_fen", "moves_opt", "moves_sub0", "depth_opt", "current_depth"])
    ds = ds.select_columns(["gameid", "root_fen", "opt_fen", "sub_fen", "moves_opt", "moves_sub0", "current_depth"])

    rows = []
    for row in tqdm.tqdm(ds.take(N), total=N):
        rows.append(row)

    df = pd.DataFrame(rows)

    df["row_id"] = np.arange(df.shape[0])
    df["moves_sub0"] = df["moves_sub0"].apply(lambda x: " ".join(x))
    df["moves_opt"] = df["moves_opt"].apply(lambda x: " ".join(x))

    df.to_parquet(p, index=False)


# if __name__ == "__main__":
#     # Initialize an empty board
#     board = chess.Board(None)
#     print("Initial empty board:")
#     print(board)

#     print(50 * "-")

#     # Setup a discovered attack scenario
#     board.set_piece_at(chess.E2, chess.Piece(chess.PAWN, chess.WHITE))  # White Pawn at e2
#     board.set_piece_at(chess.E1, chess.Piece(chess.QUEEN, chess.WHITE))  # White Queen at e1
#     board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))  # Black King at e8

#     print("Setup Board:")
#     print(board)

#     print(50 * "-")

#     # Move the pawn to e3 to reveal the queen's attack
#     move = chess.Move.from_uci("e2e3")

#     if discovered_attack(board, move):
#         print("Discovered attack detected!")
#         board.push(move)
#         print(board)
#     else:
#         print("No discovered attack detected.")

# if __name__ == "__main__":

#     save_locally(Path(__file__).parent / "train.parquet", split="train", N=200_000)
#     save_locally(Path(__file__).parent / "test.parquet", split="test", N=50_000)