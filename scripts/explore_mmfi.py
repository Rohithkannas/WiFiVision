import numpy as np

X = np.load("data/processed/mmfi/X.npy", mmap_mode="r")
Y = np.load("data/processed/mmfi/Y.npy", mmap_mode="r")

print("CSI shape :", X.shape)
print("Pose shape:", Y.shape)

print("\nFirst CSI frame:")
print(X[0])

print("\nFirst pose frame:")
print(Y[0])