from lczerolens import LczeroModel
from lczerolens.board import LczeroBoard, POLICY_INDEX

from pathlib import Path
from lczerolens import LczeroModel
import torch

ACT_LAYER = ".output/policy"

def load_lc0():
    path = str(Path(__file__).parent / "t79.onnx")
    model = LczeroModel.from_path(path).to("cuda")
    return model


model = load_lc0()
board = LczeroBoard().to_input_tensor().unsqueeze(0).to("cuda")

with torch.no_grad(), model.trace() as tracer:
    with tracer.invoke(board):
        act_module = None
        # Find the module corresponding to the activation layer.
        for name, module in model.named_modules():
            if name == ACT_LAYER:
                act_module = module
                break
        if act_module is None:
            raise ValueError("Activation layer not found in model.")

        activations = act_module.output.save()


print(activations.shape)

