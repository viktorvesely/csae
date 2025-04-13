# -------------------------Make train folder-------------------------
from pathlib import Path
from datetime import datetime
import torch
import shutil

def make_train_dir(name="experiment", path=Path("/scratch/s3799042/experiments/")):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_dir = Path(path)
    train_dir = base_dir / f"{timestamp}_{name}"
    train_dir.mkdir(parents=True, exist_ok=False)
    return train_dir


def load_checkpoint(model: torch.nn.Module, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

def save_checkpoint(model, optimizer = None, dictionary_size = None, checkpoint_path = None):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': None if optimizer is None else optimizer.state_dict(),
        'dictionary_size': dictionary_size
    }
    torch.save(checkpoint, checkpoint_path)

def copy_script_to_train_folder(train_folder, script_name="sae.py", script_name2="training.py"):
    source_path = Path(__file__).parent / script_name
    destination_path = Path(train_folder) / script_name
    shutil.copy2(source_path, destination_path)
    source_path2 = Path(__file__).parent / script_name2
    destination_path2 = Path(train_folder) / script_name2
    shutil.copy2(source_path2, destination_path2)


# --------------------Tensorboard------------------------


# import tensorflow as tf

# class TensorBoardLogger:
#     def __init__(self, train_folder):
#         self.log_dir = Path(train_folder) / "tensorboard_logs"
#         self.log_dir.mkdir(parents=True, exist_ok=True)
#         self.writer = tf.summary.create_file_writer(str(self.log_dir))

#     def __enter__(self):
#         return self

#     def log_losses(self, step, train_reconstruction_loss, train_sparsity_loss, train_contrastive_loss,
#                    valid_reconstruction_loss=None, valid_sparsity_loss=None, valid_contrastive_loss=None):
#         with self.writer.as_default():
#             tf.summary.scalar("Train/Reconstruction Loss", train_reconstruction_loss, step=step)
#             tf.summary.scalar("Train/Sparsity Loss", train_sparsity_loss, step=step)
#             tf.summary.scalar("Train/Contrastive Loss", train_contrastive_loss, step=step)

#             if valid_reconstruction_loss is not None:
#                 tf.summary.scalar("Valid/Reconstruction Loss", valid_reconstruction_loss, step=step)
#             if valid_sparsity_loss is not None:
#                 tf.summary.scalar("Valid/Sparsity Loss", valid_sparsity_loss, step=step)
#             if valid_contrastive_loss is not None:
#                 tf.summary.scalar("Valid/Contrastive Loss", valid_contrastive_loss, step=step)

#     def __exit__(self, exc_type, exc_value, traceback):
#         self.writer.close()