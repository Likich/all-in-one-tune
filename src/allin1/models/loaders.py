import torch

from typing import Optional
from omegaconf import OmegaConf
from huggingface_hub import hf_hub_download
from .allinone import AllInOne, AllInOneFinetune
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


import os

# def load_pretrained_model(
#     model_name: Optional[str] = None,
#     cache_dir: Optional[PathLike] = None,
#     device=None,
#     checkpoint_dir: Optional[PathLike] = None,
#     checkpoint_path: Optional[str] = None,  # Allow direct checkpoint path
# ):
#     if model_name in ENSEMBLE_MODELS:
#         return load_ensemble_model(model_name, cache_dir, device, checkpoint_dir)

#     model_name = model_name or list(NAME_TO_FILE.keys())[0]
#     assert model_name in NAME_TO_FILE or checkpoint_path, (
#         f"Unknown model name: {model_name} (expected one of {list(NAME_TO_FILE.keys())} or a valid checkpoint path)"
#     )

    # if device is None:
    #     device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # # Load checkpoint from custom path or predefined file
    # if checkpoint_path:
    #     print(f"Loading checkpoint from: {checkpoint_path}")
    #     checkpoint = torch.load(checkpoint_path, map_location=device)
    # else:
    #     filename = NAME_TO_FILE[model_name]
    #     if checkpoint_dir:
    #         local_checkpoint_path = os.path.join(checkpoint_dir, filename)
    #         if os.path.exists(local_checkpoint_path):
    #             print(f"Loading checkpoint from local directory: {local_checkpoint_path}")
    #             checkpoint = torch.load(local_checkpoint_path, map_location=device)
    #         else:
    #             raise FileNotFoundError(f"Checkpoint {filename} not found in {checkpoint_dir}.")
    #     else:
    #         print(f"Downloading checkpoint {filename} from Hugging Face Hub...")
    #         checkpoint_path = hf_hub_download(repo_id='taejunkim/allinone', filename=filename, cache_dir=cache_dir)
    #         checkpoint = torch.load(checkpoint_path, map_location=device)

    # # Load state_dict into the model
    # if 'state_dict' in checkpoint:
    #     print("Extracting state_dict from checkpoint...")
    #     state_dict = checkpoint['state_dict']
    # else:
    #     raise ValueError("Checkpoint does not contain a 'state_dict' key.")

    # filename = NAME_TO_FILE[model_name]
    # checkpoint_path_original = hf_hub_download(repo_id='taejunkim/allinone', filename=filename, cache_dir=cache_dir)

    # checkpoint_config = torch.load(checkpoint_path_original, map_location=device)
    # config = OmegaConf.create(checkpoint_config['config'])
    # print('config', checkpoint_config['config'])
    # config.data.num_labels = 4
    # model = AllInOne(config).to(device)
    # adjusted_state_dict = {
    #   key.replace("model.", ""): value for key, value in checkpoint["state_dict"].items()}

    # model.load_state_dict(adjusted_state_dict, strict=False)
  
    # model.eval()


    # return model


def load_pretrained_model(
    model_name: Optional[str] = None,
    cache_dir: Optional[PathLike] = None,
    device=None,
    checkpoint_dir: Optional[PathLike] = None,
    checkpoint_path: Optional[str] = None,
    num_finetune_classes: int = None,  # Number of output classes for finetuning
):
    if model_name in ENSEMBLE_MODELS:
        return load_ensemble_model(model_name, cache_dir, device, checkpoint_dir)

    model_name = model_name or list(NAME_TO_FILE.keys())[0]
    assert model_name in NAME_TO_FILE or checkpoint_path, (
        f"Unknown model name: {model_name} (expected one of {list(NAME_TO_FILE.keys())} or a valid checkpoint path)"
    )

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load checkpoint from custom path or predefined file
    if checkpoint_path:
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
    else:
        filename = NAME_TO_FILE[model_name]
        if checkpoint_dir:
            local_checkpoint_path = os.path.join(checkpoint_dir, filename)
            if os.path.exists(local_checkpoint_path):
                print(f"Loading checkpoint from local directory: {local_checkpoint_path}")
                checkpoint = torch.load(local_checkpoint_path, map_location=device)
            else:
                raise FileNotFoundError(f"Checkpoint {filename} not found in {checkpoint_dir}.")
        else:
            print(f"Downloading checkpoint {filename} from Hugging Face Hub...")
            checkpoint_path = hf_hub_download(repo_id='taejunkim/allinone', filename=filename, cache_dir=cache_dir)
            checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load state_dict from checkpoint
    if 'state_dict' in checkpoint:
        print("Extracting state_dict from checkpoint...")
        state_dict = checkpoint['state_dict']
    else:
        raise ValueError("Checkpoint does not contain a 'state_dict' key.")

    # Retrieve configuration
    filename = NAME_TO_FILE[model_name]
    checkpoint_path_original = hf_hub_download(repo_id='taejunkim/allinone', filename=filename, cache_dir=cache_dir)
    checkpoint_config = torch.load(checkpoint_path_original, map_location=device)
    print('original checkpoint is', checkpoint_path_original)
    config = OmegaConf.create(checkpoint_config['config'])
    print('config original', checkpoint_config['config'])
    checkpoint_new = torch.load(checkpoint_path)
    print('new checkpoint is', checkpoint_path)

    config.data.num_labels = 4
    model = AllInOne(config).to(device)

    # # Adjust the classifier to match the checkpoint's output shape
    # num_input_features = model.function_classifier.classifier.in_features
    # num_output_features = state_dict["function_classifier.classifier.weight"].size(0)
    # model.function_classifier.classifier = nn.Linear(num_input_features, num_output_features)

    # Load state_dict
    adjusted_state_dict = {
        key.replace("model.", ""): value for key, value in checkpoint_new["state_dict"].items()
    }
    model.load_state_dict(adjusted_state_dict, strict=False)

    model.eval()
    return model



def load_ensemble_model(
  model_name: Optional[str] = None,
  cache_dir: Optional[PathLike] = None,
  device=None,
  checkpoint_dir: Optional[PathLike] = None,
):
  models = []
  for model_name in ENSEMBLE_MODELS[model_name]:
    model = load_pretrained_model(model_name, cache_dir, device, checkpoint_dir)
    models.append(model)

  ensemble = Ensemble(models).to(device)
  ensemble.eval()

  return ensemble
