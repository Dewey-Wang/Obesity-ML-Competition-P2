import torch
import torch.nn as nn


class FlowMLP(nn.Module):

    def __init__(
        self,
        dim=512,
        cond_dim=3,
        hidden=1024
    ):
        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(dim + cond_dim + 1, hidden),
            nn.SiLU(),

            nn.Linear(hidden, hidden),
            nn.SiLU(),

            nn.Linear(hidden, hidden),
            nn.SiLU(),

            nn.Linear(hidden, dim)
        )


    def forward(
        self,
        x,
        t,
        cond
    ):

        """
        x: (B, 512)
        t: (B, 1)
        cond: (B, 3)
        """

        h = torch.cat(
            [
                x,
                cond,
                t
            ],
            dim=1
        )

        return self.net(h)