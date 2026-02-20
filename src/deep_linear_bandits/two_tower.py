from torch import nn
import torch.nn.functional as F

TEMPNORM = True
TEMP = 0.1 # 0.1 slower convergence but might be more stable; 0.05 faster convergence may be unstable

class TwoTower(nn.Module):
    def __init__(self):
        super().__init__()

        self.u_id = nn.Embedding(7176, 32)

        self.ut = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )

        self.i_id = nn.Embedding(10728, 32)

        self.it = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
    
    def forward(self, u_ids, i_ids):
        u = self.u_id(u_ids)
        i = self.i_id(i_ids)

        # u = self.ut(u)
        # i = self.it(i)

        if TEMPNORM:
            u = F.normalize(u)
            i = F.normalize(i)

            logits = (u @ i.T) / TEMP
        else:
            logits = u @ i.T

        return logits