import chess
import numpy as np
from typing import NamedTuple, List
import traceback
import math

class Scenario(NamedTuple):
    scenario_name: str
    move_index: int
    old_fen : str
    new_fen : str


class ScenarioRecord(NamedTuple):
    scenario_name: str
    move_index: int
    old_fen : str
    new_fen : str
    game_index : int

def pawn_promotion(move):
    return move.promotion is not None
    
def queen_loss(board, move, move_idx, moves):
    if (move_idx == len(moves) - 1):
      return False
    if not moves[move_idx + 1].strip():
      return False
      
    next_move = chess.Move.from_uci(moves[move_idx + 1])

    if not board.is_legal(next_move):
      return False

    captured = board.piece_at(move.to_square)

    # Check if a queen is captured
    if captured and captured.piece_type == chess.QUEEN:

        # Save the number of queens for each side
        queens_before = {
            chess.WHITE: sum(1 for p in board.piece_map().values() if p.piece_type == chess.QUEEN and p.color == chess.WHITE),
            chess.BLACK: sum(1 for p in board.piece_map().values() if p.piece_type == chess.QUEEN and p.color == chess.BLACK),
        }

        board.push(next_move)

        # Save the number of queens after the response move
        queens_after = {
            chess.WHITE: sum(1 for p in board.piece_map().values() if p.piece_type == chess.QUEEN and p.color == chess.WHITE),
            chess.BLACK: sum(1 for p in board.piece_map().values() if p.piece_type == chess.QUEEN and p.color == chess.BLACK),
        }

        # If only one queen was captured (not both), return True
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
    protected = any(True for _ in board.attackers(not board.turn, move.to_square))
    if not protected:
        return False
    
    lesser = sum(board.piece_at(s) and board.piece_at(s).piece_type in [chess.PAWN, chess.BISHOP, chess.KNIGHT] for s in board.attackers(board.turn, move.to_square))
    if lesser == 0:
      return True

    return False
    
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

            # Check for knight fork right after making the move
            if knight_fork(board, m):
                new_fen = board.fen()
                scenarios.append(Scenario("knight_fork", i, previous_fen, new_fen))
            
            if queen_loss(board, m, i, moves):
                new_fen = board.fen()
                scenarios.append(Scenario("queen_loss", i, previous_fen, new_fen))
                
            if pawn_promotion(m):
              new_fen = board.fen()
              scenarios.append(Scenario("pawn_promotion", i, previous_fen, new_fen))
              
        except Exception as e:
            print(f"Skipping invalid move {uci}: {e}")
            traceback.print_exc()
            continue

    return scenarios
        
concept_latent_activations = []
for traj_file_idx in range(25, 30):
  chess_games = np.load(f"/home3/s3799042/Trajectories/trajectory_{traj_file_idx}.npy", allow_pickle=True)
  records: list[ScenarioRecord] = []
  
  # Process each game
  for idx, game in enumerate(chess_games):
      fen = game[0]
      moves = game[1:]
      scenarios =detect_scenarios(fen, moves)
      for scenario in scenarios:
        name, at_move, old_fen, new_fen = scenario
        records.append(ScenarioRecord(name, at_move, old_fen, new_fen, idx))
  
  print(records[len(records)-1])
  

  current_file = -1
  batch_size = 300   
  activations = np.array([])
  for record in records:
    name, at_move, old_fen, new_fen, game_idx = record
    file_idx = math.ceil(game_idx / batch_size - 1)
    if file_idx > current_file:
      current_file = file_idx
      activations = np.load(f"/home3/s3799042/Trajectories/Latent/latent_{traj_file_idx}_{file_idx}.npy", allow_pickle=True)
    
    activation_idx = game_idx % batch_size
    #print(f"game idx: {game_idx} activation_idx: {activation_idx} file_idx: {file_idx} activation_len {len(activations)}")
    #print(f"activations: {len(activations[activation_idx])}")
    concept_latent_activations.append((name, at_move, game_idx, traj_file_idx, activations[activation_idx]))
  
np.save("/scratch/s3799042/data/Chess_SAE/concept_latent.npy", np.array(concept_latent_activations, dtype=object))

  
  
  
  
  