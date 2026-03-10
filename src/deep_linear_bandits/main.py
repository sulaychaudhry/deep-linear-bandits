import deep_linear_bandits.two_tower as dlb_tt
from torch.utils.data import DataLoader
import torch

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

        # Use pretrained model config & state in models/two_power.pt
        checkpoint = torch.load(dlb_tt.MODEL_PATH)
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
        pass