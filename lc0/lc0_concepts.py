import chess
import numpy as np
from typing import NamedTuple, List
import traceback
import math
import os.path
import sys

class Scenario(NamedTuple):
    scenario_name: str
    move_index: int
    starting_fen : str
    old_fen : str
    new_fen : str
    moves : str


class ScenarioRecord(NamedTuple):
    scenario_name: str
    move_index: int
    starting_fen : str
    old_fen : str
    new_fen : str
    game_index : int
    moves : str

def pawn_promotion(board, move):
    if board.piece_type_at(move.from_square) != chess.PAWN:
        return False
    

    from_rank = chess.square_rank(move.from_square)
    to_rank = chess.square_rank(move.to_square)
    if to_rank == 7 or  to_rank == 0:
      return True

    return False
    
def queen_loss(board, move, move_idx, moves):
    if (move_idx == len(moves) - 1):
      return False
    if not moves[move_idx + 1].strip():
      return False
      
    next_move = chess.Move.from_uci(moves[move_idx + 1])

    if not board.is_legal(next_move):
      return False

    captured = board.piece_at(move.to_square)

    if captured and captured.piece_type == chess.QUEEN:

        queens_before = {
            chess.WHITE: sum(1 for p in board.piece_map().values() if p.piece_type == chess.QUEEN and p.color == chess.WHITE),
            chess.BLACK: sum(1 for p in board.piece_map().values() if p.piece_type == chess.QUEEN and p.color == chess.BLACK),
        }

        board.push(next_move)

        queens_after = {
            chess.WHITE: sum(1 for p in board.piece_map().values() if p.piece_type == chess.QUEEN and p.color == chess.WHITE),
            chess.BLACK: sum(1 for p in board.piece_map().values() if p.piece_type == chess.QUEEN and p.color == chess.BLACK),
        }

        if queens_after != queens_before:
            if queens_after[chess.WHITE] == queens_before[chess.WHITE] or queens_after[chess.BLACK] == queens_before[chess.BLACK]:
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
    #protected = any(True for _ in board.attackers(not board.turn, move.to_square))
    #if not protected:
    #    return False
    
    #lesser = sum(board.piece_at(s) and board.piece_at(s).piece_type in [chess.PAWN, chess.BISHOP, chess.KNIGHT] for s in board.attackers(board.turn, move.to_square))
    #if lesser == 0:
    #  return True

    #return False
    return True
    
def detect_scenarios(fen, moves):
    scenarios = []
    board = chess.Board(fen)
    captured = {}
    
    for i, uci in enumerate(moves):
        if not uci.strip():
            continue
        try:
            previous_fen = board.fen()  
            m = chess.Move.from_uci(uci)
            if not board.is_legal(m):
                continue

            c = board.piece_at(m.to_square)
            captured[i] = c.piece_type if c else None
            board.push(m)

            if knight_fork(board, m):
              new_fen = board.fen()
              scenarios.append(Scenario("knight_fork", i, fen, previous_fen, new_fen, moves))
            
            if queen_loss(board, m, i, moves):
              new_fen = board.fen()
              scenarios.append(Scenario("queen_loss", i, fen, previous_fen, new_fen, moves))
                
            if pawn_promotion(board, m):
              new_fen = board.fen()
              scenarios.append(Scenario("pawn_promotion", i, fen, previous_fen, new_fen, moves))
              
        except Exception as e:
            print(f"Skipping invalid move {uci}: {e}")
            traceback.print_exc()
            continue

    return scenarios

def create_concept_activation_file(input_folder, input_folder_activations, output_file):
    concept_latent_activations = []
    for traj_file_idx in range(25, 30):
        chess_games = np.load(os.path.join(input_folder, f"trajectory_{traj_file_idx}.npy"), allow_pickle=True)
        records: list[ScenarioRecord] = []

        for idx, game in enumerate(chess_games):
            fen = game[0]
            moves = game[1:]
            scenarios = detect_scenarios(fen, moves)
            for scenario in scenarios:
                name, at_move, starting_fen, old_fen, new_fen, moves = scenario
                records.append(ScenarioRecord(name, at_move, starting_fen, old_fen, new_fen, idx, moves))
        
        result = combine_concepts_and_activations(output_file, input_folder_activations, records, traj_file_idx)
        concept_latent_activations.extend(result)

    np.save(output_file, np.array(concept_latent_activations, dtype=object))
    
def combine_concepts_and_activations(output_file, input_folder_activations, records, traj_file_idx):
    current_file = -1
    batch_size = 300
    activations = np.array([])
    concept_latent_activations = []
    for record in records:
        name, at_move, starting_fen, old_fen, new_fen, game_idx, moves = record
        file_idx = math.ceil(game_idx / batch_size - 1)
        if file_idx > current_file:
            current_file = file_idx
            activations = np.load(os.path.join(input_folder_activations,f"latent_{traj_file_idx}_{file_idx}.npy"),
                                  allow_pickle=True)

        activation_idx = game_idx % batch_size
        concept_latent_activations.append((name, at_move, game_idx, traj_file_idx, activations[activation_idx]))

    return concept_latent_activations


def main(arguments):
    input_folder_concepts = arguments[0]
    input_folder_activations = arguments[1]
    output_file = arguments[2]

    create_concept_activation_file(input_folder_concepts, input_folder_activations, output_file)
    
if __name__ == "__main__":
    main(sys.argv[1:])
  

  
  
  
  
  