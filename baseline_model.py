import torch
import torch.nn as nn


class CSIFrameCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(16),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2
            ),

            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2
            ),

            nn.AdaptiveAvgPool2d(
                (1, 1)
            )
        )


    def forward(self, x):

        # x:
        # (batch, 3, 114, 10)

        x = self.features(x)

        # (batch, 32, 1, 1)

        x = x.flatten(1)

        # (batch, 32)

        return x


class CSI2PoseBaseline(nn.Module):

    def __init__(self):

        super().__init__()

        self.frame_cnn = CSIFrameCNN()


        self.gru = nn.GRU(
            input_size=32,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )


        self.head = nn.Sequential(

            nn.Linear(
                64,
                128
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                34
            )
        )


    def forward(self, x):

        # ----------------------------------------------------
        # Input:
        #
        # (batch, 30, 3, 114, 10)
        # ----------------------------------------------------

        batch_size = x.shape[0]

        sequence_length = x.shape[1]


        # ----------------------------------------------------
        # Process every frame using CNN
        # ----------------------------------------------------

        x = x.reshape(
            batch_size * sequence_length,
            3,
            114,
            10
        )


        x = self.frame_cnn(x)

        # (batch * 30, 32)


        # ----------------------------------------------------
        # Restore temporal dimension
        # ----------------------------------------------------

        x = x.reshape(
            batch_size,
            sequence_length,
            32
        )


        # ----------------------------------------------------
        # GRU
        # ----------------------------------------------------

        x, _ = self.gru(x)

        # (batch, 30, 64)


        # Use final temporal representation

        x = x[:, -1, :]

        # (batch, 64)


        # ----------------------------------------------------
        # Pose prediction
        # ----------------------------------------------------

        x = self.head(x)

        # (batch, 34)

        x = x.reshape(
            batch_size,
            17,
            2
        )

        return x


# ============================================================
# TEST MODEL
# ============================================================

if __name__ == "__main__":

    model = CSI2PoseBaseline()

    x = torch.randn(
        4,
        30,
        3,
        114,
        10
    )

    y = model(x)

    print(
        "Input shape :",
        tuple(x.shape)
    )

    print(
        "Output shape:",
        tuple(y.shape)
    )

    parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        "Parameters  :",
        parameters
    )