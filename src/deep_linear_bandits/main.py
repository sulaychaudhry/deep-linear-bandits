import torch
from torch import nn
from torch.utils.data import DataLoader

from deep_linear_bandits.data import load_kuairec_big
from deep_linear_bandits.two_tower import TwoTower

from tqdm import tqdm

BATCH_SIZE = 512
TEMP = 0.1

def main():
    # Set up the train-val splitted dataset in PyTorch's Dataset format
    bm_train, bm_val = load_kuairec_big('/home/sulay/deep-linear-bandits/kuairec/data/big_matrix.csv')

    print(bm_train)
    print(bm_val)

    # Set up model on GPU
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    model = TwoTower().to(device)
    model.train() # Switch model into 'training mode'

    print(device)
    print(model)

    # Set up loss function & SGD optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters())

    print(loss_fn)
    print(optimiser)

    # Set up random sampling of batches via DataLoader
    bm_train_loader = DataLoader(bm_train, batch_size=BATCH_SIZE, shuffle=True)
    bm_val_loader = DataLoader(bm_val, batch_size=BATCH_SIZE, shuffle=True)

    # # Testing: peek at logits within this small 5-sized batch
    # # Note that logits should be highest along the diagonal since those are the pairs that it actually likes
    # # Training target should be along the diagonal (1s on diagonal and 0s elsewhere) i.e. an identity matrix
    # # CrossEntropyLoss also supports using class indices i.e. first user should have highest dot prod with first item, etc.
    # # This is easier
    # bm_train_iterator = iter(bm_train_loader)
    # test = next(bm_train_iterator)
    # print(test['user_id'][:5])
    # print(test['item_id'][:5])
    # print(res := model(test['user_id'][:5].to(device), test['item_id'][:5].to(device)))
    # print(loss_fn(res, torch.arange(0, 5, device=device))) # class index way
    # print(loss_fn(res, torch.eye(5, device=device))) # identity matrix way; result is the same

    # # Testing: ensuring that loss works as expected
    # print(loss_fn(
    #     torch.eye(5, device=device) * 10, # example in which logits are very heavily aligned towards the positive interaction
    #     torch.eye(5, device=device)
    # ))

    # Set up an epoch of training
    target = torch.arange(BATCH_SIZE, device=device) # avoid recreating tensor each time
    total_t_batches = len(bm_train_loader)
    total_v_batches = len(bm_val_loader)
    epochs = 10
    for epoch in tqdm(range(epochs)):
        avg_t_loss = 0

        model.train()
        for batch in bm_train_loader:
            optimiser.zero_grad()

            preds = model(batch['user_id'].to(device), batch['item_id'].to(device)) / TEMP

            if preds.size(dim=0) == BATCH_SIZE:
                loss = loss_fn(preds, target)
            else:
                loss = loss_fn(preds, torch.arange(preds.size(dim=0), device=device))

            loss.backward()
            optimiser.step()

            avg_t_loss += loss.item()
            
        avg_t_loss /= total_t_batches

        # Evaluate loss on the validation set too as a sanity check
        model.eval()
        avg_v_loss = 0
        with torch.no_grad():  
            for batch in bm_val_loader:
                preds = model(batch['user_id'].to(device), batch['item_id'].to(device)) / TEMP

                if preds.size(dim=0) == BATCH_SIZE:
                    loss = loss_fn(preds, target)
                else:
                    loss = loss_fn(preds, torch.arange(preds.size(dim=0), device=device))

                avg_v_loss += loss.item()

        avg_v_loss /= total_v_batches
        
        tqdm.write(f"End of epoch {epoch}")
        tqdm.write(f"Average training loss per-batch: {avg_t_loss}")
        tqdm.write(f"Average validation loss per-batch: {avg_v_loss}")
