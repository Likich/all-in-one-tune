import numpy as np
import torch
from ..typings import AllInOneOutput, Segment
from ..config import Config, HARMONIX_LABELS
from .helpers import local_maxima, peak_picking, event_frames_to_time


def postprocess_functional_structure(
    logits: AllInOneOutput,
    cfg: Config,
):
    # Compute probabilities for sections and functions
    raw_prob_sections = torch.sigmoid(logits.logits_section[0])
    raw_prob_functions = torch.softmax(logits.logits_function[0], dim=0)
    prob_sections, _ = local_maxima(raw_prob_sections, filter_size=4 * cfg.min_hops_per_beat + 1)
    prob_sections = prob_sections.cpu().numpy()
    prob_functions = raw_prob_functions.cpu().numpy()

    # Identify boundary candidates using peak picking
    boundary_candidates = peak_picking(
        boundary_activation=prob_sections,
        window_past=12 * cfg.fps,
        window_future=12 * cfg.fps,
    )
    boundary = boundary_candidates > 0.0

    # Handle case where no boundaries are found
    if len(boundary) == 0:
        print("Warning: No boundaries detected.")
        return []  # Return an empty list of segments

    # Compute duration and boundary times
    duration = len(prob_sections) * cfg.hop_size / cfg.sample_rate
    pred_boundary_times = event_frames_to_time(boundary, cfg)

    # Add start and end times if necessary
    if len(pred_boundary_times) == 0:
        pred_boundary_times = np.array([0, duration])  # Default to the full duration
    else:
        if pred_boundary_times[0] != 0:
            pred_boundary_times = np.insert(pred_boundary_times, 0, 0)
        if pred_boundary_times[-1] != duration:
            pred_boundary_times = np.append(pred_boundary_times, duration)

    pred_boundaries = np.stack([pred_boundary_times[:-1], pred_boundary_times[1:]]).T

    # Compute segment functions and labels
    pred_boundary_indices = np.flatnonzero(boundary)
    prob_segment_function = np.split(prob_functions, pred_boundary_indices, axis=1)
    pred_labels = [p.mean(axis=1).argmax().item() for p in prob_segment_function]

    # Build segments
    segments = []
    for (start, end), label in zip(pred_boundaries, pred_labels):
        segment = Segment(
            start=start,
            end=end,
            label=HARMONIX_LABELS[label],
        )
        segments.append(segment)

    return segments
