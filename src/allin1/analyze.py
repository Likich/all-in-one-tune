import torch

from typing import List, Union
from tqdm import tqdm
from .demix import demix
from .spectrogram import extract_spectrograms
from .models import load_pretrained_model
from .visualize import visualize as _visualize
from .sonify import sonify as _sonify
from .helpers import (
    run_inference,
    expand_paths,
    check_paths,
    rmdir_if_empty,
    save_results,
)
from .utils import mkpath, load_result
from .typings import AnalysisResult, PathLike


def analyze(
    paths: Union[PathLike, List[PathLike]],
    out_dir: PathLike = None,
    visualize: Union[bool, PathLike] = False,
    sonify: Union[bool, PathLike] = False,
    model: str = "harmonix-fold2",  # Default to harmonix-fold2
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    checkpoint_path: PathLike = None,  # Path to your custom checkpoint
    include_activations: bool = False,
    include_embeddings: bool = False,
    demix_dir: PathLike = "./demix",
    spec_dir: PathLike = "./spec",
    keep_byproducts: bool = False,
    overwrite: bool = False,
    multiprocess: bool = True,
) -> Union[AnalysisResult, List[AnalysisResult]]:
    """
    Analyzes the provided audio files and returns the analysis results.
    """
    return_list = True
    if not isinstance(paths, list):
        return_list = False
        paths = [paths]
    if not paths:
        raise ValueError("At least one path must be specified.")
    paths = [mkpath(p) for p in paths]
    paths = expand_paths(paths)
    check_paths(paths)
    demix_dir = mkpath(demix_dir)
    spec_dir = mkpath(spec_dir)

    if out_dir is None or overwrite:
        todo_paths = paths
        exist_paths = []
    else:
        out_paths = [mkpath(out_dir) / path.with_suffix(".json").name for path in paths]
        todo_paths = [path for path, out_path in zip(paths, out_paths) if not out_path.exists()]
        exist_paths = [out_path for path, out_path in zip(paths, out_paths) if out_path.exists()]

    print(f"=> Found {len(exist_paths)} tracks already analyzed and {len(todo_paths)} tracks to analyze.")
    if exist_paths:
        print("=> To re-analyze, please use --overwrite option.")

    results = []
    if exist_paths:
        results += [
            load_result(
                exist_path,
                load_activations=include_activations,
                load_embeddings=include_embeddings,
            )
            for exist_path in tqdm(exist_paths, desc="Loading existing results")
        ]

    if todo_paths:
        demix_paths = demix(todo_paths, demix_dir, device)
        spec_paths = extract_spectrograms(demix_paths, spec_dir, multiprocess)

        print(f"=> Initializing model: {model}")
        pretrained_model = load_pretrained_model(model_name=model, device=device)
        pretrained_model.to(device)  # Ensure model is on the specified device

        cfg = pretrained_model.cfg
        cfg.data.num_labels = 4  # Update number of labels
        pretrained_model.function_classifier.classifier = torch.nn.Linear(
            cfg.data.num_instruments * cfg.dim_embed, 4
        ).to(device)  # Ensure classifier is on the correct device
        print(f"=> Updated classification head to output {cfg.data.num_labels} labels.")

        if checkpoint_path:
            print(f"=> Loading weights from checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            pretrained_model.load_state_dict(checkpoint["state_dict"], strict=False)
            print("=> Checkpoint weights loaded.")

        with torch.no_grad():
            pbar = tqdm(zip(todo_paths, spec_paths), total=len(todo_paths))
            for path, spec_path in pbar:
                pbar.set_description(f"Analyzing {path.name}")

                # Ensure spectrogram is loaded to the correct device
                spec = torch.load(spec_path).to(device)

                result = run_inference(
                    path=path,
                    spec_path=spec,
                    model=pretrained_model,
                    device=device,
                    include_activations=include_activations,
                    include_embeddings=include_embeddings,
                )

                if out_dir is not None:
                    save_results(result, out_dir)

                results.append(result)

    results = sorted(results, key=lambda result: paths.index(result.path))

    if visualize:
        if visualize is True:
            visualize = "./viz"
        _visualize(results, out_dir=visualize, multiprocess=multiprocess)
        print(f"=> Plots are successfully saved to {visualize}")

    if sonify:
        if sonify is True:
            sonify = "./sonif"
        _sonify(results, out_dir=sonify, multiprocess=multiprocess)
        print(f"=> Sonified tracks are successfully saved to {sonify}")

    if not keep_byproducts:
        for path in demix_paths:
            for stem in ["bass", "drums", "other", "vocals"]:
                (path / f"{stem}.wav").unlink(missing_ok=True)
            rmdir_if_empty(path)
        rmdir_if_empty(demix_dir / "htdemucs")
        rmdir_if_empty(demix_dir)

        for path in spec_paths:
            path.unlink(missing_ok=True)
        rmdir_if_empty(spec_dir)

    if not return_list:
        return results[0]
    return results

