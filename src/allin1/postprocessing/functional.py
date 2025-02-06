import numpy as np
import torch
from ..typings import AllInOneOutput, Segment
from ..config import Config, HARMONIX_LABELS
from .helpers import local_maxima, peak_picking, event_frames_to_time

def postprocess_functional_structure(
    logits: AllInOneOutput,
    cfg: Config,
):
    print("\n=== Debugging postprocess_functional_structure ===")

    # Compute probabilities for sections and functions
    raw_prob_sections = torch.sigmoid(logits.logits_section[0])
    raw_prob_functions = torch.softmax(logits.logits_function[0], dim=0)

    print(f"Raw probabilities for sections (logits_section): {raw_prob_sections.shape}")
    print(f"Raw probabilities for functions (logits_function): {raw_prob_functions.shape}")

    prob_sections, _ = local_maxima(raw_prob_sections, filter_size=4 * cfg.min_hops_per_beat + 1)
    prob_sections = prob_sections.cpu().numpy()
    prob_functions = raw_prob_functions.cpu().numpy()

    print(f"Processed prob_sections shape: {prob_sections.shape}")
    print(f"Processed prob_functions shape: {prob_functions.shape}")

    # Identify boundary candidates using peak picking
    boundary_candidates = peak_picking(
        boundary_activation=prob_sections,
        window_past=12 * cfg.fps,
        window_future=12 * cfg.fps,
    )
    boundary = boundary_candidates > 0.0

    print(f"Boundary candidates: {boundary_candidates}")
    print(f"Boundary (after thresholding): {boundary}")

    if len(boundary) == 0:
        print("Warning: No boundaries detected.")
        return []  # Return an empty list of segments

    # Compute duration and boundary times
    duration = len(prob_sections) * cfg.hop_size / cfg.sample_rate
    pred_boundary_times = event_frames_to_time(boundary, cfg)

    print(f"Predicted boundary times: {pred_boundary_times}")

    # Add start and end times if necessary
    if len(pred_boundary_times) == 0:
        print("No boundaries detected, defaulting to full duration.")
        pred_boundary_times = np.array([0, duration])  # Default to the full duration
    else:
        if pred_boundary_times[0] != 0:
            pred_boundary_times = np.insert(pred_boundary_times, 0, 0)
        if pred_boundary_times[-1] != duration:
            pred_boundary_times = np.append(pred_boundary_times, duration)

    print(f"Final boundary times after adjustments: {pred_boundary_times}")

    pred_boundaries = np.stack([pred_boundary_times[:-1], pred_boundary_times[1:]]).T
    print(f"Predicted boundaries shape: {pred_boundaries.shape}")

    # Compute segment functions and labels
    pred_boundary_indices = np.flatnonzero(boundary)
    print(f"Predicted boundary indices: {pred_boundary_indices}")

    if len(pred_boundary_indices) == 0:
        print("No boundary indices found, skipping function assignment.")
        return []

    prob_segment_function = np.split(prob_functions, pred_boundary_indices, axis=1)

    if not prob_segment_function:
        print("Error: No function probabilities were split. Check peak picking output.")
        return []

    pred_labels = [p.mean(axis=1).argmax().item() for p in prob_segment_function]

    print(f"Predicted labels (index format): {pred_labels}")
    print(f"Label mapping: {[HARMONIX_LABELS[label] for label in pred_labels]}")

    # Build segments
    segments = []
    for (start, end), label in zip(pred_boundaries, pred_labels):
        segment = Segment(
            start=start,
            end=end,
            label=HARMONIX_LABELS[label],
        )
        segments.append(segment)

    print(f"Final segments: {segments}")
    print("=== End of Debugging ===\n")

    return segments
