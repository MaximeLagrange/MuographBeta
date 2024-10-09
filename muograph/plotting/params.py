from typing import Tuple

# Number of bins for histograms
n_bins: int = 50

# Histogram transparency
alpha: float = 0.6

# Distance unit
d_unit: str = "mm"

# Matplotlib default font
fontsize: int = 13
labelsize: int = 12
titlesize: int = 16

fontweigh: str = "normal"
font = {"weight": "normal", "size": fontsize, "family": "Serif"}

# Matplotlib colormaps
cmap: str = "jet"

# Histogram XY ratio
xy_golden_ratio: float = 1.4
hist_scale = 4.5
hist_figsize: Tuple[float, float] = (xy_golden_ratio * hist_scale, 1 * hist_scale)

# Scale of matplotlib figures with subplots
# Size of figures with subplots is defined as
# (scale * ncols * xy_ratio[0], scale * nrows * xy_ratio[1])
scale: int = 3

# 2D histogram
n_bins_2D = 100
