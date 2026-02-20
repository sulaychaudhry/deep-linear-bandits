from deep_linear_bandits.data import load_kuairec_big
from deep_linear_bandits.two_tower import TwoTower
import torch
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm

BATCH_SIZE=1024 # 512 I know works; but 1024 is good on GPU

def main():
    bm_train, bm_val = load_kuairec_big()

    #print(bm.user_ids)
    #print(bm.item_ids)

    print(len(bm_train))

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    model = TwoTower().to(device)

    #print(model)

    bm_t_ldr = DataLoader(bm_train, batch_size=BATCH_SIZE, shuffle=True)
    bm_v_ldr = DataLoader(bm_val, batch_size=BATCH_SIZE, shuffle=True)

    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters())

    epochs = 100
    for epoch in range(epochs):
        model.train()
        l_t = 0
        for batch in tqdm(bm_t_ldr):
            opt.zero_grad()

            logits = model(batch['user_id'].to(device), batch['item_id'].to(device))
            target = torch.arange(logits.size(dim=0), device=device)

            loss = loss_fn(logits, target)
            loss.backward()
            opt.step()

            l_t += loss.item()
        l_t /= len(bm_t_ldr)

        model.eval()
        l_v = 0
        with torch.no_grad():
            for batch in tqdm(bm_v_ldr):
                logits = model(batch['user_id'].to(device), batch['item_id'].to(device))
                target = torch.arange(logits.size(dim=0), device=device)
                l_v += loss_fn(logits, target).item()
        l_v /= len(bm_v_ldr)

        print(f"End of training epoch {epoch}; avg batch loss (train): {l_t}")
        print(f"End of training epoch {epoch}; avg batch loss (val): {l_v}")

        



    # # # ----------------------
    # # Experiment to gauge self convergence
    # # (turns out model needs temp=0.05 to 0.1 to make small dissimilarities/angular differences matter more)
    # # (this ensures that it can actually learn to separate them, as softmax thinks everything is roughly equal before, especially in just the range -1 to 1 which is what L2 norm will give)

    # batch = next(iter(bm_ldr))

    # logits = model(batch['user_id'].to(device), batch['item_id'].to(device))
    # #print(logits)
    # #print(torch.argmax(logits, dim=1))

    # loss_fn = nn.CrossEntropyLoss()
    # opt = torch.optim.Adam(model.parameters(), lr=0.003) # 0.003 observed to be good so far

    # #print(loss_fn(logits, torch.arange(logits.size(dim=0), device=device)).item())

    # print(f"User IDs: {list(enumerate(batch['user_id']))}")
    # print(f"Item IDs: {list(enumerate(batch['item_id']))}")

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