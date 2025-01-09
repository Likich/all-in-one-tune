import torch

from typing import Optional
from omegaconf import OmegaConf
from huggingface_hub import hf_hub_download
from .allinone import AllInOne
from .ensemble import Ensemble
from ..typings import PathLike
import tempfile

NAME_TO_FILE = {
  'harmonix-fold0': 'harmonix-fold0-0vra4ys2.pth',
  'harmonix-fold1': 'harmonix-fold1-3ozjhtsj.pth',
  'harmonix-fold2': 'harmonix-fold2-gmgo0nsy.pth',
  'harmonix-fold3': 'harmonix-fold3-i92b7m8p.pth',
  'harmonix-fold4': 'harmonix-fold4-1bql5qo0.pth',
  'harmonix-fold5': 'harmonix-fold5-x4z5zeef.pth',
  'harmonix-fold6': 'harmonix-fold6-x7t226rq.pth',
  'harmonix-fold7': 'harmonix-fold7-qwwskhg6.pth',
}

ENSEMBLE_MODELS = {
  'harmonix-all': [
    'harmonix-fold0',
    'harmonix-fold1',
    'harmonix-fold2',
    'harmonix-fold3',
    'harmonix-fold4',
    'harmonix-fold5',
    'harmonix-fold6',
    'harmonix-fold7',
  ],
}


def load_pretrained_model(
    model_name: Optional[str] = None,
    device=None,
):
    # Always use 'harmonix-fold2' or any other model as default
    model_name = "harmonix-fold2"

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Use a temporary directory for caching to avoid reusing cached models
    with tempfile.TemporaryDirectory() as temp_cache_dir:
        filename = NAME_TO_FILE[model_name]
        checkpoint_path = hf_hub_download(
            repo_id="taejunkim/allinone",
            filename=filename,
            cache_dir=temp_cache_dir  # Temporary directory ensures no caching
        )

        checkpoint = torch.load(checkpoint_path, map_location=device)
        config = OmegaConf.create(checkpoint["config"])

        model = AllInOne(config).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

    return model


def load_ensemble_model(
  model_name: Optional[str] = None,
  cache_dir: Optional[PathLike] = None,
  device=None,
):
  models = []
  for model_name in ENSEMBLE_MODELS[model_name]:
    model = load_pretrained_model(model_name, cache_dir, device)
    models.append(model)

  ensemble = Ensemble(models).to(device)
  ensemble.eval()

  return ensemble
