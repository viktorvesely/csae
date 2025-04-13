import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, TensorDataset
import numpy as np
from sklearn.metrics import confusion_matrix
from collections import Counter
import os.path
import sys

def get_class_indices(labels):
  classes, class_indices = np.unique(labels, return_inverse=True)
  return class_indices
  
def get_latent_activatsions_per_concept(path_concept_latent):
  concept_latent_activations = np.load(path_concept_latent, allow_pickle=True) 

  X = np.array([sample[4] for sample in concept_latent_activations])
  X = X.reshape(len(X), -1)

  Y = np.array([sample[0] for sample in concept_latent_activations])
  Y = get_class_indices(Y)

  Y_knight_fork = np.where(Y == 0)[0]
  Y_pawn_promotion = np.where(Y == 1)[0]
  Y_queen_loss = np.where(Y == 2)[0]

  X_knight_fork = X[Y_knight_fork]
  X_pawn_promotion = X[Y_pawn_promotion]
  X_queen_loss = X[Y_queen_loss]
  return X_knight_fork, X_queen_loss

def get_most_overlapping_indices(X_knight_fork, X_queen_loss):
  top_indices_per_row_knight = np.argsort(X_knight_fork, axis=1)[:, -20:]
  top_indices_per_row_queen = np.argsort(X_queen_loss, axis=1)[:, -20:]

  flattened_indices_knight = top_indices_per_row_knight.flatten()
  flattened_indices_queen = top_indices_per_row_queen.flatten()

  counts_knight = np.bincount(flattened_indices_knight, minlength=len(X_knight_fork))
  counts_queen = np.bincount(flattened_indices_queen, minlength=len(X_queen_loss))

  most_overlapping_indices_knight = np.argsort(counts_knight)[-20:]
  most_overlapping_indices_queen = np.argsort(counts_queen)[-20:]
  return most_overlapping_indices_knight, most_overlapping_indices_queen

def print_results(most_overlapping_indices_knight, most_overlapping_indices_queen):
  print("Most overlapping indices knight:", np.flip(most_overlapping_indices_knight))

  print("Most overlapping indices queen:", np.flip(most_overlapping_indices_queen))

  for indice in most_overlapping_indices_queen:
    if indice in most_overlapping_indices_knight:
      most_overlapping_indices_knight = np.delete(most_overlapping_indices_knight, np.where(most_overlapping_indices_knight == indice))

  print("Most overlapping indices knight with indices queen removed:", most_overlapping_indices_knight)

def main(arguments):
  path_concept_latent = arguments[0]
  X_knight_fork, X_queen_loss = get_latent_activatsions_per_concept(path_concept_latent)
  most_overlapping_indices_knight, most_overlapping_indices_queen = get_most_overlapping_indices(X_knight_fork, X_queen_loss)
  print_results(most_overlapping_indices_knight, most_overlapping_indices_queen)

if __name__ == "__main__":
    main(sys.argv[1:])