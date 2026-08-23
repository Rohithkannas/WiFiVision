import zipfile
import scipy.io as sio
import io
import numpy as np
import matplotlib.pyplot as plt

ZIP_PATH = "data/raw/E01.zip"

with zipfile.ZipFile(ZIP_PATH) as z:
    filename = next(
        n for n in z.namelist()
        if "wifi-csi/frame" in n and n.endswith(".mat")
    )

    data = sio.loadmat(
        io.BytesIO(z.read(filename))
    )

amplitude = data["CSIamp"]
phase = data["CSIphase"]

fig, axes = plt.subplots(2, 1, figsize=(10, 10))

im1 = axes[0].imshow(
    amplitude[0],
    aspect="auto"
)
axes[0].set_title("Raw CSI Amplitude - Channel 1")
axes[0].set_xlabel("Dimension 3")
axes[0].set_ylabel("Subcarrier")
fig.colorbar(im1, ax=axes[0], label="Amplitude")

im2 = axes[1].imshow(
    phase[0],
    aspect="auto",
    cmap="twilight"
)
axes[1].set_title("Raw CSI Phase - Channel 1")
axes[1].set_xlabel("Dimension 3")
axes[1].set_ylabel("Subcarrier")
fig.colorbar(im2, ax=axes[1], label="Phase (radians)")

plt.tight_layout()
plt.show()