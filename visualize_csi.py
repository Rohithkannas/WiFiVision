import numpy as np
import matplotlib.pyplot as plt

X = np.load("data/processed/mmfi/X.npy", mmap_mode="r")

sample = X[0]

fig, axes = plt.subplots(3, 1, figsize=(10, 12))

for i in range(3):
    im = axes[i].imshow(sample[i], aspect="auto", cmap="viridis")
    axes[i].set_title(f"CSI Channel {i + 1}")
    axes[i].set_xlabel("Dimension 3")
    axes[i].set_ylabel("Subcarrier")
    fig.colorbar(im, ax=axes[i], label="Normalized CSI Amplitude")

plt.tight_layout()
plt.show()