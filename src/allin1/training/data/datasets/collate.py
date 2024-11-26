import numpy as np
import torch
from collections import defaultdict


def collate_fn(raw_batch):
    variable_length_batch = defaultdict(list)
    
    for row in raw_batch:
        for key, value in list(row.items()):
            if isinstance(value, list):
                variable_length_batch[key].append(row.pop(key))
    
    max_T = max(x['spec'].shape[1] for x in raw_batch)  # Get the maximum sequence length
    
    batch = []
    for raw_data in raw_batch:
        data = {}
        for key, value in raw_data.items():
            # Allow raw strings or numpy arrays to pass through
            if key in ['track_key', 'true_bpm', 'widen_true_bpm', 'true_bpm_int']:
                # Ensure `true_bpm_int` is converted to a tensor
                if key == 'true_bpm_int' and isinstance(value, pd.Series):
                    value = value.to_numpy()
                if isinstance(value, np.ndarray):
                    value = torch.tensor(value, dtype=torch.float32)
                data[key] = value
            
            # Handle fixed-length sequence data
            elif key in [
                'true_beat', 'true_downbeat', 'true_section', 'true_function',
                'widen_true_beat', 'widen_true_downbeat', 'widen_true_section',
            ]:
                data[key] = value[:max_T]  # Truncate to max_T
            
            # Handle spectrograms
            elif key in ['spec']:
                T = raw_data[key].shape[1]
                spec = raw_data[key]
                if T < max_T:
                    spec = np.pad(spec, ((0, 0), (0, max_T - T), (0, 0)), 'constant')  # Pad with zeros
                    mask = np.pad(np.ones(T), (0, max_T - T), 'constant')  # Create a mask
                else:
                    mask = np.ones(max_T)
                data[key] = torch.tensor(spec, dtype=torch.float32)  # Convert to tensor
                data['mask'] = torch.tensor(mask, dtype=torch.float32)
            
            # Handle unknown keys
            else:
                raise ValueError(f'Unknown key: {key}')
        
        batch.append(data)

    # Debugging: Check the batch structure
    print("Before collate:", batch)
    
    # Ensure the batch is compatible with `default_collate`
    batch = torch.utils.data.default_collate(batch)
    batch = {**batch, **variable_length_batch}  # Merge variable-length batch data
    
    # Debugging: Check the final collated batch
    print("After collate:", batch)
    
    return batch
