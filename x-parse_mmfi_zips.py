"""Parse MM-Fi WiFi-CSI directly from the E0x.zip files (no extraction).

Builds:
    X.npy         [N, 3, 114, 10]  CSI amplitude
    Y.npy         [N, 17, 2]       normalized 2D pose
    subj.npy      [N]              subject ID
    act.npy       [N]              action ID
    split_xsubj.npy[N]             cross-subject split

Reads:
    CSIamp (3, 114, 10) per frame
    ground_truth.npy (F, 17, 3) per action
"""

import zipfile
import io
import os
import sys
import glob
import time
import numpy as np
import scipy.io as scio


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else "assets/MM-Fi"
OUT = sys.argv[2] if len(sys.argv) > 2 else "aether-arena/staging/mmfi_npy"

os.makedirs(OUT, exist_ok=True)

ZIPS = sorted(
    glob.glob(
        os.path.join(DATA_DIR, "E0*.zip")
    )
)

print("zips:", [os.path.basename(z) for z in ZIPS], flush=True)


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def action_key(name):
    """
    Example:
        E01/S01/A01/wifi-csi/frame001.mat

    Returns:
        E01/S01/A01
    """
    parts = name.split("/")
    return "/".join(parts[:3])


def fix_csi(a):
    """
    Clean and min-max normalize one CSI frame.
    """
    a = np.asarray(a, np.float32)

    # Replace infinities with NaN
    a[np.isinf(a)] = np.nan

    # Replace NaN with mean
    if np.isnan(a).any():
        mean_value = np.nanmean(a)

        if np.isnan(mean_value):
            mean_value = 0.0

        a[np.isnan(a)] = mean_value

    # Min-max normalization
    rng = a.max() - a.min()

    if rng > 0:
        return (a - a.min()) / rng

    return np.zeros_like(a)


# ---------------------------------------------------------
# Pass 1: Count frames
# ---------------------------------------------------------

total = 0
plans = []

for zp in ZIPS:

    z = zipfile.ZipFile(zp)

    frames = {}
    gts = {}

    for name in z.namelist():

        # CSI frame
        if "wifi-csi/frame" in name and name.endswith(".mat"):
            key = action_key(name)

            frames.setdefault(key, []).append(name)

        # Ground truth
        elif name.endswith("ground_truth.npy"):
            key = action_key(name)

            gts[key] = name

    count = sum(
        len(frame_list)
        for key, frame_list in frames.items()
        if key in gts
    )

    total += count

    plans.append((zp, frames, gts))

    z.close()

    print(
        f"{os.path.basename(zp)}: {count} frames",
        flush=True
    )


print(f"TOTAL frames: {total}", flush=True)


# ---------------------------------------------------------
# Allocate arrays
# ---------------------------------------------------------

X = np.zeros(
    (total, 3, 114, 10),
    dtype=np.float32
)

Y = np.zeros(
    (total, 17, 2),
    dtype=np.float32
)

SUBJ = np.zeros(
    total,
    dtype=np.int64
)

ACT = np.zeros(
    total,
    dtype=np.int64
)


# ---------------------------------------------------------
# Pass 2: Read CSI + pose
# ---------------------------------------------------------

i = 0

t0 = time.time()

for zp, frames, gts in plans:

    z = zipfile.ZipFile(zp)

    for ak in sorted(frames):

        if ak not in gts:
            continue

        # Subject ID
        sid = int(
            [
                p for p in ak.split("/")
                if p.startswith("S")
            ][0][1:]
        )

        # Action ID
        aid = int(
            [
                p for p in ak.split("/")
                if p.startswith("A")
            ][0][1:]
        )

        # Ground truth
        gt = np.load(
            io.BytesIO(
                z.read(gts[ak])
            )
        )

        gt = (
            gt
            .reshape(-1, 17, 3)
           [:, :, :2]
            .astype(np.float32)
        )

        # Process every CSI frame
        for fn in sorted(frames[ak]):

            # frame001 → index 0
            idx = (
                int(
                    fn.split("frame")[-1]
                    .split(".")[0]
                )
                - 1
            )

            if idx < 0 or idx >= len(gt):
                continue

            try:

                csi = scio.loadmat(
                    io.BytesIO(
                        z.read(fn)
                    )
                )["CSIamp"]

            except Exception:
                continue

            # Store data
            X[i] = fix_csi(
                np.asarray(csi, np.float32)
            )

            Y[i] = gt[idx]

            SUBJ[i] = sid

            ACT[i] = aid

            i += 1

            # Progress
            if i and i % 20000 < 300:

                elapsed = (
                    time.time() - t0
                ) / 60

                print(
                    f"  {i}/{total} "
                    f"({elapsed:.1f} min)",
                    flush=True
                )

    z.close()


# ---------------------------------------------------------
# Trim unused allocation
# ---------------------------------------------------------

X = X[:i]
Y = Y[:i]
SUBJ = SUBJ[:i]
ACT = ACT[:i]


# ---------------------------------------------------------
# Normalize pose coordinates
# ---------------------------------------------------------

for d in range(2):

    lo = Y[:, :, d].min()
    hi = Y[:, :, d].max()

    Y[:, :, d] = (
        (Y[:, :, d] - lo)
        / (hi - lo + 1e-6)
    )


# ---------------------------------------------------------
# Save processed data
# ---------------------------------------------------------

np.save(
    f"{OUT}/X.npy",
    X
)

np.save(
    f"{OUT}/Y.npy",
    Y
)

np.save(
    f"{OUT}/subj.npy",
    SUBJ
)

np.save(
    f"{OUT}/act.npy",
    ACT
)

np.save(
    f"{OUT}/split_xsubj.npy",
    (SUBJ > 30).astype(np.int64)
)


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

print(
    f"DONE: X{X.shape} "
    f"Y{Y.shape} "
    f"saved to {OUT}",
    flush=True
)

print(
    f"cross-subject: "
    f"train(S01-30)={int((SUBJ <= 30).sum())} "
    f"test(S31-40)={int((SUBJ > 30).sum())}",
    flush=True
)

print(
    f"unique subjects: {np.unique(SUBJ)}",
    flush=True
)

print(
    f"unique actions: {np.unique(ACT)}",
    flush=True
)