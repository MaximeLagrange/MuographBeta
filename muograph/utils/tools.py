import torch


def normalize(s: torch.tensor) -> torch.tensor:
    s_max, s_min = torch.max(s), torch.min(s)

    return (s - s_min) / (s_max - s_min)
