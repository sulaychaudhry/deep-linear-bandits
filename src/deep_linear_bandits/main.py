import deep_linear_bandits.two_tower as dlb_tt
import torch
from deep_linear_bandits.data import KRSmall

USE_PRETRAINED = True

def main():
    # Set up a device for PyTorch to use the GPU (if available)
    device = (
        torch.accelerator.current_accelerator()
        if torch.accelerator.is_available()
        else torch.device('cpu')
    )

    model: dlb_tt.TwoTower
    if USE_PRETRAINED:
        print("Loading pre-trained two-tower model...")

        # Use pretrained model config & state in models/two_tower.pt
        checkpoint = torch.load(dlb_tt.MODEL_PATH, map_location=device)
        model = dlb_tt.TwoTower(**checkpoint["model_config"]).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.compile()
        model.eval()

        print("Two-tower model loaded!")
    else:
        # Create & train a two-tower model over many epochs
        # This method tracks the training & validation loss each epoch
        model = dlb_tt.generate_two_tower_model(device)
        model.eval()
    
    # Use inference mode for quickest use of the model after training
    # Begin the process of freezing this two-tower model & using the embeddings for the bandit
    with torch.inference_mode():
        # Retrieve data for the small matrix
        small_matrix = KRSmall()

        # Compute all user & item embeddings for the small matrix
        user_embeddings = model.user_tower(
            *small_matrix.tower_ready_users(device)
        )
        item_embeddings = model.item_tower(
            *small_matrix.tower_ready_items(device)
        )

        

        # 1. process krsmall -> user IDs, item IDs, binary label
        # 2. for all of those users & items, I need to retrieve features so they can be embedded
        # 3. randomise user stream:
        # ---- user comes in, check what items they can actually be served
        # ---- for those items, apply all bandit policies
        # ---- return results from bandit policies
        # ---- observe rewards
        # 4. track observed reward, then perform next interaction; all part of the simulator
        # 5. simulator returns results
