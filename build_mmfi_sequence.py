import zipfile
import io
import os
import re
import numpy as np
import scipy.io as scio


# ============================================================
# CONFIGURATION
# ============================================================

ZIP_PATH = "data/raw/E01.zip"
OUT_DIR = "data/processed/mmfi_sequence"

os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# FIND ACTION SEQUENCES
# ============================================================

print("Opening:", ZIP_PATH)

with zipfile.ZipFile(ZIP_PATH, "r") as z:

    names = z.namelist()

    # Find every action sequence containing WiFi CSI
    actions = sorted(
        set(
            "/".join(n.split("/")[:3])
            for n in names
            if "/wifi-csi/frame" in n
            and n.endswith(".mat")
        )
    )

    print("Action sequences:", len(actions))


    # ========================================================
    # DETERMINE SEQUENCE SIZE
    # ========================================================

    first_action = actions[0]

    frame_names = [
        n for n in names
        if n.startswith(first_action + "/wifi-csi/frame")
        and n.endswith(".mat")
    ]

    frame_names = sorted(
        frame_names,
        key=lambda n: int(
            re.search(r"frame(\d+)\.mat$", n).group(1)
        )
    )

    num_frames = len(frame_names)

    print("Frames per action:", num_frames)


    # ========================================================
    # INSPECT FIRST FRAME
    # ========================================================

    first_data = scio.loadmat(
        io.BytesIO(z.read(frame_names[0]))
    )

    amplitude_shape = first_data["CSIamp"].shape
    phase_shape = first_data["CSIphase"].shape

    print("CSI amplitude shape:", amplitude_shape)
    print("CSI phase shape:", phase_shape)


    # ========================================================
    # DATASET DIMENSIONS
    # ========================================================

    num_sequences = len(actions)

    num_antennas = amplitude_shape[0]
    num_subcarriers = amplitude_shape[1]
    num_time = amplitude_shape[2]

    # Expected:
    # sequences = 270
    # frames    = 297
    # CSI       = 3 x 114 x 10


    # ========================================================
    # ALLOCATE ARRAYS
    # ========================================================

    amplitude = np.zeros(
        (
            num_sequences,
            num_frames,
            num_antennas,
            num_subcarriers,
            num_time
        ),
        dtype=np.float32
    )

    phase = np.zeros_like(amplitude)

    pose = np.zeros(
        (
            num_sequences,
            num_frames,
            17,
            2
        ),
        dtype=np.float32
    )

    subject = np.zeros(
        num_sequences,
        dtype=np.int64
    )

    action = np.zeros(
        num_sequences,
        dtype=np.int64
    )


    # ========================================================
    # READ EACH ACTION SEQUENCE
    # ========================================================

    for seq_idx, action_key in enumerate(actions):

        parts = action_key.split("/")

        # Example:
        # E01 / S01 / A01

        subject_id = int(
            parts[1][1:]
        )

        action_id = int(
            parts[2][1:]
        )

        subject[seq_idx] = subject_id
        action[seq_idx] = action_id


        # ----------------------------------------------------
        # CSI frames
        # ----------------------------------------------------

        frame_names = [
            n for n in names
            if n.startswith(action_key + "/wifi-csi/frame")
            and n.endswith(".mat")
        ]

        frame_names = sorted(
            frame_names,
            key=lambda n: int(
                re.search(
                    r"frame(\d+)\.mat$",
                    n
                ).group(1)
            )
        )


        # ----------------------------------------------------
        # Ground truth
        # ----------------------------------------------------

        gt_name = action_key + "/ground_truth.npy"

        gt = np.load(
            io.BytesIO(
                z.read(gt_name)
            )
        )

        gt = gt.reshape(
            -1,
            17,
            3
        )[:, :, :2].astype(
            np.float32
        )


        # ----------------------------------------------------
        # Read every frame
        # ----------------------------------------------------

        for frame_idx, frame_name in enumerate(frame_names):

            if frame_idx >= len(gt):
                break

            try:

                data = scio.loadmat(
                    io.BytesIO(
                        z.read(frame_name)
                    )
                )

                amplitude[
                    seq_idx,
                    frame_idx
                ] = data["CSIamp"]

                phase[
                    seq_idx,
                    frame_idx
                ] = data["CSIphase"]

                pose[
                    seq_idx,
                    frame_idx
                ] = gt[frame_idx]

            except Exception as e:

                print(
                    "Error:",
                    frame_name,
                    e
                )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            (seq_idx + 1) % 10 == 0
            or seq_idx == 0
            or seq_idx == num_sequences - 1
        ):

            print(
                f"Processed "
                f"{seq_idx + 1}/{num_sequences} "
                f"sequences"
            )


    # ========================================================
    # SAVE
    # ========================================================

    print("\nSaving...")

    np.save(
        os.path.join(
            OUT_DIR,
            "amplitude.npy"
        ),
        amplitude
    )

    np.save(
        os.path.join(
            OUT_DIR,
            "phase.npy"
        ),
        phase
    )

    np.save(
        os.path.join(
            OUT_DIR,
            "pose.npy"
        ),
        pose
    )

    np.save(
        os.path.join(
            OUT_DIR,
            "subject.npy"
        ),
        subject
    )

    np.save(
        os.path.join(
            OUT_DIR,
            "action.npy"
        ),
        action
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n======================================")
print("MM-Fi sequence dataset created")
print("======================================")

print("Amplitude :", amplitude.shape)
print("Phase     :", phase.shape)
print("Pose      :", pose.shape)
print("Subject   :", subject.shape)
print("Action    :", action.shape)

print("\nOutput:")
print(OUT_DIR)