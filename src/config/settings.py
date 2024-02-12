import os
import torch

ROOT_PATH =  os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
DATA_PATH =  os.path.join(ROOT_PATH, "data")
MODELS_PATH =  os.path.join(ROOT_PATH, "models")
RESULTS_PATH =  os.path.join(ROOT_PATH, "results")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TORCH_GENERATOR = torch.Generator().manual_seed(42)