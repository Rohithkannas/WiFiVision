import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from baseline_dataset import MMFiBaselineDataset
from baseline_model import CSI2PoseBaseline


# ============================================================
# CONFIGURATION
# ============================================================

SPLIT_DIR = "data/processed/baseline_split"

CHECKPOINT_DIR = "models"
CHECKPOINT_PATH = os.path.join(
    CHECKPOINT_DIR,
    "baseline_best.pt"
)

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)

BATCH_SIZE = 8
EPOCHS = 30
LEARNING_RATE = 1e-3

PATIENCE = 5

DEVICE = torch.device("cpu")


# ============================================================
# LOAD SPLITS
# ============================================================

train_idx = np.load(
    os.path.join(
        SPLIT_DIR,
        "train_idx.npy"
    )
)

val_idx = np.load(
    os.path.join(
        SPLIT_DIR,
        "val_idx.npy"
    )
)

test_idx = np.load(
    os.path.join(
        SPLIT_DIR,
        "test_idx.npy"
    )
)


# ============================================================
# DATASETS
# ============================================================

train_dataset = MMFiBaselineDataset(
    train_idx,
    window_size=30,
    stride=15
)

val_dataset = MMFiBaselineDataset(
    val_idx,
    window_size=30,
    stride=15
)

test_dataset = MMFiBaselineDataset(
    test_idx,
    window_size=30,
    stride=15
)


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# MODEL
# ============================================================

model = CSI2PoseBaseline().to(
    DEVICE
)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING
# ============================================================

best_val_loss = float("inf")
patience_counter = 0


print("========================================")
print("MM-Fi BASELINE TRAINING")
print("========================================")

print(
    "Device:",
    DEVICE
)

print(
    "Train windows:",
    len(train_dataset)
)

print(
    "Validation windows:",
    len(val_dataset)
)

print(
    "Test windows:",
    len(test_dataset)
)

print(
    "Batch size:",
    BATCH_SIZE
)

print(
    "Epochs:",
    EPOCHS
)

print()


for epoch in range(1, EPOCHS + 1):

    # ========================================================
    # TRAIN
    # ========================================================

    model.train()

    train_loss = 0.0

    for x, y in train_loader:

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()

        prediction = model(x)

        loss = criterion(
            prediction,
            y
        )

        loss.backward()

        optimizer.step()

        train_loss += (
            loss.item()
            * x.size(0)
        )

    train_loss /= len(
        train_dataset
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    val_loss = 0.0

    with torch.no_grad():

        for x, y in val_loader:

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            prediction = model(x)

            loss = criterion(
                prediction,
                y
            )

            val_loss += (
                loss.item()
                * x.size(0)
            )

    val_loss /= len(
        val_dataset
    )


    # ========================================================
    # PRINT
    # ========================================================

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.6f} | "
        f"Val Loss: {val_loss:.6f}"
    )


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if val_loss < best_val_loss:

        best_val_loss = val_loss
        patience_counter = 0

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict":
                    model.state_dict(),
                "optimizer_state_dict":
                    optimizer.state_dict(),
                "val_loss":
                    val_loss
            },
            CHECKPOINT_PATH
        )

        print(
            "  -> Best model saved"
        )

    else:

        patience_counter += 1

        print(
            f"  -> No improvement "
            f"({patience_counter}/{PATIENCE})"
        )


    # ========================================================
    # EARLY STOPPING
    # ========================================================

    if patience_counter >= PATIENCE:

        print(
            "\nEarly stopping."
        )

        break


# ============================================================
# FINAL
# ============================================================

print("\n========================================")
print("TRAINING COMPLETE")
print("========================================")

print(
    "Best validation loss:",
    best_val_loss
)

print(
    "Checkpoint:",
    CHECKPOINT_PATH
)