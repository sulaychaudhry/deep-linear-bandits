import click
import torch
from datetime import datetime
import os
import shutil
import math

import deep_linear_bandits.data as dlb_data
import deep_linear_bandits.two_tower as dlb_tt

# Set up device for PyTorch to use the GPU (if available)
device = (
    torch.accelerator.current_accelerator()
    if torch.accelerator.is_available()
    else torch.device('cpu')
)

@click.group()
def cli() -> None:
    """
    The command-line interface for training the two-tower model and running the bandit simulations.
    """

    print(f"PyTorch device: {device}")

@cli.command('train-tt')
@click.option('--save-name', type=str, default="%d-%m-%Y_%H-%M-%S.%f")
@click.option('--side-features/--no-side-features', default=True)
def train_tt(
    save_name: str,
    use_side_features: bool
) -> None:
    """
    Interface for training the two-tower model.
    """

    # Set up the directory for saving this model & its metrics
    path = f'tt-models/{save_name}'
    if os.path.exists(path): shutil.rmtree(path)
    os.makedirs(path)

    # Get training & validation (positive) user-item interactions from KuaiRec-Big
    pos_intrs_train, pos_intrs_val = dlb_data.preprocess_krbig_interactions()

    # Get user side features: The categorical and numeric user features, alongside the size of each categorical user feature
    (user_cat_feats, user_cat_sizes), user_numeric_feats = dlb_data.preprocess_user_features()

    # Convert categorical feature sizes to embedding dimensions for each feature
    user_cat_emb_sizes = [
        # Square root preferred over fourth root due to relatively small vocab sizes for each feature - fourth root would be too aggressive
        # Capped at 16 to prevent any from too heavily dominating the representation vs. user ID
        min(math.ceil(math.sqrt(size), 16)) for size in user_cat_sizes
    ]
    
    # Get item side features: the item categories
    item_categories = dlb_data.preprocess_item_categories()
    
    # Create two-tower model & move to GPU
    # For simplicity of implementation, side feature setup is still passed (trivially) but will be ignored in model construction if use_side_features=False; this does not affect results at all
    model = dlb_tt.TwoTower(
        # User side features
        user_cat_input_sizes=user_cat_sizes,
        user_cat_emb_sizes=user_cat_emb_sizes,
        user_num_numeric_features=user_numeric_feats.shape[1],

        # Item side features
        num_item_categories=item_categories.shape[1],

        use_side_features=use_side_features
    ).to(device)

    




import deep_linear_bandits.two_tower as dlb_tt
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
    # Compute all user & item embeddings for the small matrix using the two-tower model
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

    # Pass small matrix data to the simulator for bandit simulation
    simulator = Simulator(
        device,
        small_matrix,
        user_embeddings,
        item_embeddings
    )

    simulator.run()