from torch import nn
import torch.nn.functional as F

TEMPNORM = True

class TwoTower(nn.Module):
    def __init__(self):
        super().__init__()

        self.u_id = nn.Embedding(7176, 32)
        self.i_id = nn.Embedding(10728, 32)
    
    def forward(self, u_ids, i_ids):
        if TEMPNORM:
            TEMP = 0.05 # 0.1 slower convergence but might be more stable; 0.05 faster convergence may be unstable

            u = F.normalize(self.u_id(u_ids))
            i = F.normalize(self.i_id(i_ids))

            logits = (u @ i.T) / TEMP
        else:
            u = self.u_id(u_ids)
            i = self.i_id(i_ids)

            logits = u @ i.T

        return logits