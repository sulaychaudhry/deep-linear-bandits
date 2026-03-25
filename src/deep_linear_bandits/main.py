import click
import torch
from datetime import datetime
import os
import shutil
import math
import pickle

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
@click.option(
    '--save-name',
    type=str,
    default="%d-%m-%Y_%H-%M-%S.%f",
    show_default=True,
    help='Directory name under tt-models/ used to save the model & its results.'
)
@click.option(
    '--side-features/--no-side-features',
    default=True,
    show_default=True,
    help='Enable or disable user/item side features in both towers.'
)
@click.option(
    '--hidden-size',
    type=click.IntRange(1),
    multiple=True,
    default=(128,),
    show_default=True,
    help='Repeat for each hidden layer, e.g. --hidden-size 256 --hidden-size 128'
)
@click.option(
    '--relu/--no-relu',
    default=True,
    show_default=True,
    help='Enable/disable the use of ReLU in the towers to learn non-linearity.'
)
@click.option(
    '--dropout',
    type=float,
    type=click.FloatRange(min=0.0, max=1.0)
    default=0.2,
    show_default=True,
    help='Set training dropout probability to reduce tower overfitting; set to 0.0 to disable dropout.'
)
@click.option(
    '--l2-norm/--no-l2-norm',
    default=True,
    show_default=True,
    help='Enable or disable L2 normalisation on user/item tower outputs - useful to coerce the model to learn cosine similarity instead i.e. meaningful orientation instead of "cheating" via large magnitudes.'
)
@click.option(
    '--logit-temp',
    type=click.FloatRange(min=0.0, min_open=True),
    default=0.07,
    show_default=True,
    help='Divisor used to scale similarity scores (logits) prior to softmax; smaller values make the model "sharper" i.e. it increases its confidence in small similarity differences.'
)
def train_tt(
    save_name: str,
    side_features: bool,
    hidden_sizes: tuple[int, ...],
    relu: bool,
    dropout: float,
    l2_norm: bool,
    logit_temp: float
) -> None:
    """
    Interface for training the two-tower model.
    """

    # Set up the directory for saving this model & its metrics
    path = f'tt-models/{save_name}/'
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
        min(math.ceil(math.sqrt(size)), 16) for size in user_cat_sizes
    ]
    
    # Get item side features: the item categories
    item_categories = dlb_data.preprocess_item_categories()
    
    # Set up arguments to pass to the TwoTower constructor; these are saved in a dictionary to ensure that the model can be reloaded after it's saved
    model_args = {
        # User side features
        "user_cat_input_sizes": user_cat_sizes,
        "user_cat_emb_sizes": user_cat_emb_sizes,
        "user_num_numeric_features": user_numeric_feats.shape[1],

        # Item side features
        "num_item_categories": item_categories.shape[1],

        # Whether to actually use these side features or not
        # (They're passed trivially anyway for simplicity & consistency of implementation, but ignored with no effect on model quality)
        "use_side_features": side_features,

        # The hidden layer sizes & output embedding widths
        "hidden_sizes": list(hidden_sizes),
        "output_size": 64,

        # Extra tower settings
        "use_relu": relu,
        "dropout": dropout,
        "use_l2_norm": l2_norm,
        "logit_temp": logit_temp
    }

    print("Arguments passed to model constructor:")
    for param, arg in model_args.items():
        print("    " + param + ": " + str(arg))

    # Save the model args for reconstructing the model later
    with open(path + 'model_args.pkl', 'wb') as f:
        pickle.dump(model_args, f)

    # Create two-tower model & move to GPU
    # + compile model for faster forward & backward passes
    model = dlb_tt.TwoTower(**model_args).to(device)
    model.compile()

    

    




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