import deep_linear_bandits.two_tower as dlb_tt
from torch.utils.data import DataLoader
import torch

def main():
    # Set up a device for PyTorch to use the GPU (if available)
    device = (
        torch.accelerator.current_accelerator()
        if torch.accelerator.is_available()
        else torch.device('cpu')
    )

    # Create & train a two-tower model over many epochs
    # This method tracks the training & validation loss each epoch
    model = dlb_tt.generate_two_tower_model(device)