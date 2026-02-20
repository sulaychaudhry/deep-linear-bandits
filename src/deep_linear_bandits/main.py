from deep_linear_bandits.data import load_kuairec_big
from deep_linear_bandits.two_tower import TwoTower
import torch
from torch.utils.data import DataLoader
from torch import nn

def main():
    bm = load_kuairec_big()

    #print(bm.user_ids)
    #print(bm.item_ids)

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    model = TwoTower().to(device)
    model.train()

    #print(model)

    bm_ldr = DataLoader(bm, batch_size=32, shuffle=True)

    

    # # ----------------------
    # Experiment to gauge self convergence
    # (turns out model needs temp=0.05 to 0.1 to make small dissimilarities/angular differences matter more)
    # (this ensures that it can actually learn to separate them, as softmax thinks everything is roughly equal before, especially in just the range -1 to 1 which is what L2 norm will give)

    # batch = next(iter(bm_ldr))

    # logits = model(batch['user_id'].to(device), batch['item_id'].to(device))
    # #print(logits)
    # #print(torch.argmax(logits, dim=1))

    # loss_fn = nn.CrossEntropyLoss()
    # opt = torch.optim.Adam(model.parameters(), lr=0.003)

    # #print(loss_fn(logits, torch.arange(logits.size(dim=0), device=device)).item())

    # print(f"User IDs: {batch['user_id']}")
    # print(f"Item IDs: {batch['item_id']}")

    # for i in range(1000):
    #     opt.zero_grad()

    #     #print(batch['user_id'])
    #     #print(batch['item_id'])

    #     logits = model(batch['user_id'].to(device), batch['item_id'].to(device))

    #     #print(logits)
    #     #print(torch.max(logits, dim=1))

    #     target = torch.arange(logits.size(dim=0), device=device)
    #     #print(target)

    #     loss = loss_fn(logits, target)

    #     loss.backward()
    #     opt.step()
        
    #     if i % 100 == 0:
    #         print("-----------------------")
    #         print(f"Iteration {i}: loss seen {loss.item()}")
    #         print(f"Max item logits per user: {torch.max(logits,dim=1)}")
    #         # Should look for the mischosen items; check what the logits are for that user's row vs. actual item

    #     #print(f"Loss seen: {loss.item()}")
    
    # # -------------------------------------------