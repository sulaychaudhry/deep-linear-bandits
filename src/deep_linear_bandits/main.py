import deep_linear_bandits.two_tower as dlb_tt
import torch
from deep_linear_bandits.data import KRSmall
from deep_linear_bandits.simulator import Simulator

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

        # Pass small matrix data & embeddings to the simulator for bandit simulation
        simulator = Simulator(
            device,
            small_matrix,
            user_embeddings,
            item_embeddings
        )

        simulator.run()