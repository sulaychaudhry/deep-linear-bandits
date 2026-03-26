import click
import torch
from torch.utils.data import DataLoader
from datetime import datetime
import os
import shutil
import math
import pickle
import json

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
    default=datetime.now().strftime("%d-%m-%Y_%H-%M-%S.%f"),
    show_default=True,
    help='Directory name under tt-models/ used to save the model & its results.'
)
@click.option(
    '--watch-threshold',
    type=click.FloatRange(0.0, 5.0),
    default=2.0,
    show_default=True,
    help='The minimum watch_ratio to observe to classify a given video as a "positive" interaction.'
)
@click.option(
    '--metric-k',
    type=click.IntRange(1),
    multiple=True,
    default=(10, 50, 100, 200),
    show_default=True,
    help='Repeat for each K value to use for computing validation Recall@K and NDCG@K, e.g. --metric-k 10 --metric-k 50 ...'
)
@click.option(
    '--best-k',
    type=click.IntRange(1),
    default=50,
    show_default=True,
    help='The K value to compute Recall@K for in freezing the model at its best performance; must be a passed metric-k argument.'
)
@click.option(
    '--side-features/--no-side-features',
    default=True,
    show_default=True,
    help='Enable or disable user/item side features in both towers.'
)
@click.option(
    # Useful for emulating a viable MF baseline if side features disabled and --id-emb-dims set to e.g. 64
    '--skip-towers',
    is_flag=True,
    help='For debugging & evaluation purposes, this disables the tower functionality (skips their MLPs) and purely treats the concatenated latent embeddings as suitable L2-normalisable output embeddings. Note that output embedding dimensions are not guaranteed to match --output-size as a result, nor are necessarily the same width suitable for dot prod unless --no-side-features is passed.'
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
    '--output-size',
    type=click.IntRange(1),
    default=64,
    show_default=True,
    help='The final output dimensions of a user/item embedding after the two-tower network.'
)
@click.option(
    '--relu/--no-relu',
    default=True,
    show_default=True,
    help='Enable/disable the use of ReLU in the towers to learn non-linearity.'
)
@click.option(
    '--dropout',
    type=click.FloatRange(min=0.0, max=1.0),
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
@click.option(
    '--batch-size',
    type=click.IntRange(1),
    default=1024,
    show_default=True,
    help='Batch size to use for training the two-tower model.'
)
@click.option(
    '--epochs',
    type=click.IntRange(1),
    default=100,
    show_default=True,
    help='Epochs to train the two-tower model for.'
)
@click.option(
    '--num-negatives',
    type=click.IntRange(1),
    default=256,
    show_default=True,
    help='The number of uniform negatives to sample per positive interaction (only used with --negative-sampling uniform).'
)
@click.option(
    '--negative-sampling',
    type=click.Choice(('uniform', 'in-batch')),
    default='uniform',
    show_default=True,
    help='Negative sampling strategy: "uniform" samples K random items per batch; "in-batch" uses other positives in the batch as negatives.'
)
@click.option(
    '--lr',
    type=click.FloatRange(min=0.0, min_open=True),
    default=0.001,
    show_default=True,
    help='Learning rate for the optimiser used to train the model.'
)
@click.option(
    '--optimiser',
    type=click.Choice(('adam', 'adamw')),
    default='adam',
    show_default=True,
    help='Optimiser to use: "adam" (no weight decay) or "adamw" (decoupled weight decay).'
)
@click.option(
    '--data-workers',
    type=click.IntRange(1),
    default=4,
    show_default=True,
    help='Number of workers (concurrent CPU threads) to use for dispatching training & validation batches, each.'
)
@click.option(
    '--id-emb-dims',
    type=click.IntRange(1),
    default=32, # 32 default since KuaiRec is relatively small
    show_default=True,
    help='The number of dimensions to embed user & item IDs to, prior to passing them through their respective towers.'
)
@click.option(
    '--item-cat-emb-dims',
    type=click.IntRange(1),
    default=16, # 31 item categories, each item has 4 categories maximally; 8-wide can also be appropriate
    show_default=True,
    help='The number of dimensions to embed an item\'s "categories" side feature to.'
)
@click.option(
    '--user-cat-emb-root',
    type=click.Choice(('2', '4')),
    # Square root preferred over fourth root due to relatively small vocab sizes for each feature - fourth root would be too aggressive
    default='2',
    show_default=True,
    help='Root (square or quartic) to use for determining embedding sizes for each of a user\'s categorical side features from their vocabulary size.'
)
@click.option(
    '--user-cat-emb-cap',
    type=click.IntRange(1),
    # Capped at 16 to prevent any from too heavily dominating the representation vs. user ID
    default=16,
    show_default=True,
    help='Maximum size that any individual user categorical side feature can have for its intermediate embedding (use to prevent side features from dominating the ID embedding).'
)
def train_tt(
    save_name: str,
    watch_threshold: float,
    metric_k: tuple[int, ...],
    best_k: int,
    side_features: bool,
    skip_towers: bool,
    hidden_size: tuple[int, ...],
    output_size: int,
    relu: bool,
    dropout: float,
    l2_norm: bool,
    logit_temp: float,
    batch_size: int,
    epochs: int,
    num_negatives: int,
    negative_sampling: str,
    lr: float,
    optimiser: str,
    data_workers: int,
    id_emb_dims: int,
    item_cat_emb_dims: int,
    user_cat_emb_root: int,
    user_cat_emb_cap: int
) -> None:
    """
    Interface for training the two-tower model.
    """

    flags = locals()
    if best_k not in metric_k:
        raise click.BadParameter(f"Flag best-k={str(best_k)} must be one of the available metric-k={str(metric_k)}", param_hint='--best-k')
    if skip_towers and side_features:
        raise click.BadOptionUsage("Debug (experimental) flag --skip-towers is enabled but without --no-side-features; user and item embedding widths are mismatched and will not be suitable for computing dot-product similarities.")

    # Set up the directory for saving this model & its metrics
    path = f'tt-models/{save_name}/'
    if os.path.exists(path): shutil.rmtree(path)
    os.makedirs(path)

    print(f"\ntrain_tt flags (logged to {path}flags.txt):")
    with open(path + 'flags.txt', 'a') as f:
        for flag_name, flag_arg in flags.items():
            print(f"    {flag_name}: {str(flag_arg)}")
            f.write(f"{flag_name}: {str(flag_arg)}\n")

    # Get training & validation (positive) user-item interactions from KuaiRec-Big
    pos_intrs_train, pos_intrs_val = dlb_data.preprocess_krbig_interactions(watch_threshold)

    # Get user side features: The categorical and numeric user features, alongside the size of each categorical user feature
    (user_cat_feats, user_cat_sizes), user_numeric_feats = dlb_data.preprocess_user_features()

    # Convert categorical feature sizes to embedding dimensions for each feature
    power = 1 / int(user_cat_emb_root)
    user_cat_emb_sizes = [
        min(math.ceil(size ** power), user_cat_emb_cap) for size in user_cat_sizes
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

        # Debug/eval option: disable towers entirely, just use nn.Embeddings all concatenated
        "skip_towers": skip_towers,

        # Sizes of intermediate representations
        "id_emb_dims": id_emb_dims,
        "item_cat_emb_dims": item_cat_emb_dims,

        # Whether to actually use these side features or not
        # (They're passed trivially anyway for simplicity & consistency of implementation, but ignored with no effect on model quality)
        "use_side_features": side_features,

        # The hidden layer sizes & output embedding widths
        "hidden_sizes": list(hidden_size),
        "output_size": output_size,

        # Extra tower settings
        "use_relu": relu,
        "dropout": dropout,
        "use_l2_norm": l2_norm,
        "logit_temp": logit_temp
    }

    print(f"\nArguments passed to model constructor (saved to {path}model_args.pkl):")
    for param, arg in model_args.items():
        print("    " + param + ": " + str(arg))
    print()

    # Save the model args for reconstructing the model later
    with open(path + 'model_args.pkl', 'wb') as f:
        pickle.dump(model_args, f)

    # Create two-tower model & move to GPU
    # + compile model for faster forward & backward passes
    model = dlb_tt.TwoTower(**model_args).to(device)
    model.compile()

    # Pass training & validation datasets through KRBig for PyTorch batching compatibility
    training_set = dlb_data.KRBig(
        pos_intrs_train, 
        user_cat_feats, 
        user_numeric_feats, 
        item_categories
    )
    validation_set = dlb_data.KRBig(
        pos_intrs_val, 
        user_cat_feats, 
        user_numeric_feats, 
        item_categories
    )

    # Set up DataLoaders for dispatching training & validation batches
    # Use multithreading & pinned memory (between RAM & CUDA) for much quicker retrieval
    train_loader = DataLoader(
        training_set,
        batch_size=batch_size,
        shuffle=True, # Shuffle per epoch to reduce it from fitting on data order
        num_workers=data_workers,
        pin_memory=(True if torch.accelerator.is_available() else False),
        persistent_workers=True
    )
    val_loader = DataLoader(
        validation_set,
        batch_size=batch_size,
        shuffle=False, # Don't shuffle per-epoch for validation, not necessary & has been shuffled during data split
        num_workers=data_workers,
        pin_memory=(True if torch.accelerator.is_available() else False),
        persistent_workers=True
    )

    # Train the two-tower model; returns per-epoch metrics for later visualisation
    metrics, model = dlb_tt.train_two_tower(
        model=model,
        device=device,

        train_loader=train_loader,
        val_loader=val_loader,

        training_set=training_set,
        validation_set=validation_set,

        item_categories=item_categories,
        user_cat_feats=user_cat_feats,
        user_numeric_feats=user_numeric_feats,

        epochs=epochs,
        num_negatives=num_negatives,
        negative_sampling=negative_sampling,
        lr=lr,
        optimiser=(
            torch.optim.Adam(model.parameters(), lr=lr)
            if optimiser=='adam' else
            torch.optim.AdamW(model.parameters(), lr=lr)
        )
    )

    # Save metrics for later visualisation
    with open(path + 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"\nModel metrics have been saved to {path}metrics.json")

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