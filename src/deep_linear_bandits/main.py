import click
import torch
from torch.utils.data import DataLoader
from datetime import datetime
import os
import shutil
import math
import pickle
import json
import numpy as np
import glob

import deep_linear_bandits.data as dlb_data
import deep_linear_bandits.two_tower as dlb_tt
import deep_linear_bandits.simulator as dlb_sim
import deep_linear_bandits.plot as dlb_plot

# Get DLB_DIR from environment instead; can export this from the slurm scripts
DLB_DIR = os.environ.get("DLB_DIR", os.getcwd())
if not DLB_DIR.endswith("/"):
    DLB_DIR += "/"
DATA_DIR = DLB_DIR + "kuairec/data/"

# Set up device for PyTorch to use the GPU (if available)
# (& use TF32 speedup if available)
torch.set_float32_matmul_precision('high')
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

    if not os.path.exists(DATA_DIR):
        raise Exception(f"The KuaiRec dataset must be located at {DATA_DIR}")

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
    default=(256, 128),
    show_default=True,
    help='Repeat for each hidden layer, e.g. --hidden-size 256 --hidden-size 128'
)
@click.option(
    '--output-size',
    type=click.IntRange(1),
    default=32,
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
    default=20,
    show_default=True,
    help='The number of uniform negatives to sample per positive interaction (only used with --negative-sampling uniform/score-weighted/watch-ratio).'
)
@click.option(
    '--negative-sampling',
    type=click.Choice(('uniform', 'in-batch', 'score-weighted', 'watch-ratio')),
    default='uniform',
    show_default=True,
    help='Negative sampling strategy: "uniform" samples K random items per batch; "in-batch" uses other positives in the batch as negatives; "score-weighted" samples each user\'s K negatives proportional to current model scores (hard negative mining relative to model current embedding space); "watch-ratio" uses the user\'s watch ratio for a video to place it into a negative hardness band, with different bands having different sampling probabilities (hard negative mining relative to how little the user watched the video).'
)
@click.option(
    '--wr-band-ratio',
    nargs=5,
    type=click.FloatRange(min=0.0),
    default=(0.35, 0.3, 0.2, 0.1, 0.05),
    show_default=True,
    help='watch_ratio band sampling ratios for `--negative-sampling watch-ratio`: (UNSEEN, [0, T/4), [T/4, T/2), [T/2, 3T/4), [3T/4, T)) where T is the watch threshold; e.g. (1, 1, 1, 1, 1) is uniform.'
)
@click.option(
    '--score-sharpness',
    type=click.FloatRange(min=0.0),
    default=1.0,
    show_default=True,
    help='Sharpness of score-weighted negative sampling. 0 = uniform; higher values increasingly concentrate random sampling on items the model currently scores highly for each user. Only used with --negative-sampling score-weighted.'
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
@click.option(
    '--seed',
    type=int,
    default=int(np.random.default_rng().integers(0, 2**63)),
    help='RNG seed for reproducibility. If omitted, a random seed is generated and logged.'
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
    wr_band_ratio: tuple[float, ...],
    score_sharpness: float,
    lr: float,
    optimiser: str,
    data_workers: int,
    id_emb_dims: int,
    item_cat_emb_dims: int,
    user_cat_emb_root: int,
    user_cat_emb_cap: int,
    seed: int
) -> None:
    """
    Interface for training the two-tower model.
    """

    flags = locals()
    if best_k not in metric_k:
        raise click.BadParameter(f"Flag best-k={str(best_k)} must be one of the available metric-k={str(metric_k)}", param_hint='--best-k')
    if skip_towers and side_features:
        raise click.BadOptionUsage("Debug (experimental) flag --skip-towers is enabled but without --no-side-features; user and item embedding widths are mismatched and will not be suitable for computing dot-product similarities.")
    
    if all(r == 0.0 for r in wr_band_ratio):
        raise click.BadParameter(f"Flag wr-band-ratio={str(wr_band_ratio)}: at least one value must be non-zero.", param_hint='--wr-band-ratio')

    # Set up the directory for saving this model & its metrics
    path = DLB_DIR + f'tt-models/{save_name}/'
    if os.path.exists(path): shutil.rmtree(path)
    os.makedirs(path)

    # Save all flags so that 'simulate' can retrieve e.g. watch_threshold, output_size
    # & display the flags to the user
    with open(path + 'flags.json', 'w') as f:
        json.dump(flags, f, indent=4)
    print(f"\ntrain_tt flags (logged to {path}flags.json):")
    for flag_name, flag_arg in flags.items():
        print(f"    {flag_name}: {str(flag_arg)}")

    # Get training & validation interactions from KuaiRec-Big
    all_intrs_train, all_intrs_val = dlb_data.preprocess_krbig_interactions(DATA_DIR, watch_threshold)
    pos_intrs_train = all_intrs_train[all_intrs_train["watch_ratio"] >= watch_threshold].drop(columns=["watch_ratio"])
    pos_intrs_val = all_intrs_val[all_intrs_val["watch_ratio"] >= watch_threshold].drop(columns=["watch_ratio"])

    # Get user side features: The categorical and numeric user features, alongside the size of each categorical user feature
    (user_cat_feats, user_cat_sizes), user_numeric_feats = dlb_data.preprocess_user_features(DATA_DIR)

    # Convert categorical feature sizes to embedding dimensions for each feature
    power = 1 / int(user_cat_emb_root)
    user_cat_emb_sizes = [
        min(math.ceil(size ** power), user_cat_emb_cap) for size in user_cat_sizes
    ]
    
    # Get item side features: the item categories
    item_categories = dlb_data.preprocess_item_categories(DATA_DIR)

    # Build per-user per-item sampling weight matrices for watch-ratio negative sampling;
    # val weights mask out training positives to avoid validation loss falsely ballooning
    # from training positives
    if negative_sampling == 'watch-ratio':
        train_wr_weights = dlb_data.build_wr_weight_matrix(
            all_intrs_train, wr_band_ratio, watch_threshold
        ).to(device)
        val_wr_weights = dlb_data.build_wr_weight_matrix(
            all_intrs_val, wr_band_ratio, watch_threshold,
            mask_user=pos_intrs_train["user_id"].to_numpy(),
            mask_item=pos_intrs_train["video_id"].to_numpy()
        ).to(device)
    else:
        train_wr_weights = val_wr_weights = None

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

    # Set RNG seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)

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
    dl_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        training_set,
        batch_size=batch_size,
        shuffle=True, # Shuffle per epoch to reduce it from fitting on data order
        num_workers=data_workers,
        pin_memory=(True if torch.accelerator.is_available() else False),
        persistent_workers=True,
        generator=dl_generator # Needed since shuffle=True
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

        metric_ks=list(metric_k),
        best_k=best_k,

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
        score_sharpness=score_sharpness,
        train_wr_weights=train_wr_weights,
        val_wr_weights=val_wr_weights,
        optimiser=(
            torch.optim.Adam(model.parameters(), lr=lr)
            if optimiser=='adam' else
            torch.optim.AdamW(model.parameters(), lr=lr)
        )
    )

    # Save model weights (state_dict) so simulate can load them later
    torch.save(model.state_dict(), path + 'model.pt')
    print(f"\nModel weights have been saved to {path}model.pt")

    # Save metrics
    with open(path + 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Model metrics have been saved to {path}metrics.json")

    # Visualise metrics & save to disk
    dlb_tt.visualise(
        metrics,
        path + 'metrics.png',
        list(metric_k),

        training_set,
        validation_set
    )
    print(f"Plotted metrics have been saved to {path}metrics.png")

@cli.command('simulate')
@click.option(
    '--save-name',
    type=str,
    default=datetime.now().strftime("%d-%m-%Y_%H-%M-%S.%f"),
    show_default=True,
    help='Directory name under simulations/ used to save results & plots.'
)
@click.option(
    '--model',
    'model_name',
    type=str,
    required=True,
    help='Directory name under tt-models/ of the trained two-tower model to load.'
)
@click.option(
    '--hadamard/--no-hadamard',
    default=True,
    show_default=True,
    help='Include/exclude the element-wise (Hadamard) product of user & item embeddings in context vectors.'
)
@click.option(
    '--watch-threshold',
    type=click.FloatRange(0.0, 5.0),
    default=None,
    help='Minimum watch_ratio to classify an interaction as positive. If omitted, read from the model\'s flags.json.'
)
@click.option(
    '--epsilon',
    type=float,
    multiple=True,
    default=(0.01, 0.05, 0.1, 0.2),
    show_default=True,
    help='Repeat for each epsilon value for ε-greedy policies, e.g. --epsilon 0.05 --epsilon 0.1'
)
@click.option(
    '--alpha',
    type=float,
    multiple=True,
    default=(0.1, 0.5, 1.0, 2.0, 5.0),
    show_default=True,
    help='Repeat for each alpha value for LinUCB policies, e.g. --alpha 0.5 --alpha 1.0'
)
@click.option(
    '--ts-v',
    type=float,
    multiple=True,
    default=(0.25, 0.5, 1.0, 2.0, 5.0),
    show_default=True,
    help='Repeat for each variance scale value for Thompson Sampling policies, e.g. --ts-v 0.5 --ts-v 1.0'
)
@click.option(
    '--lambda', 'lmbda',
    type=click.FloatRange(min=0.0, min_open=True),
    default=1.0,
    show_default=True,
    help='Ridge regularisation lambda for all linear bandit policies (LinUCB, ε-greedy, Thompson Sampling).'
)
@click.option(
    '--rounds',
    type=click.IntRange(1),
    default=10000,
    show_default=True,
    help='Number of simulation rounds.'
)
@click.option(
    '--seed-count',
    type=click.IntRange(1),
    default=1,
    show_default=True,
    help='Total number of independent seeds to run (averaged for final plots/metrics).'
)
@click.option(
    '--seed',
    type=int,
    default=int(np.random.default_rng().integers(0, 2**63)),
    help='RNG seed for full reproducibility. If omitted, a random seed is generated and logged.'
)
@click.option(
    '--seed-index',
    type=click.IntRange(0),
    default=None,
    help='Run only this one seed (0-indexed out of --seed-count). Intended for parallelising seeds across separate Slurm jobs; use "dlb collate" afterwards to merge results.'
)
@click.option(
    '--longtail-percentile',
    type=click.FloatRange(0.0, 100.0),
    default=80.0,
    show_default=True,
    help='Popularity percentile threshold to use when computing long-tail item coverage.'
)
@click.option(
    '--metric-interval',
    type=click.IntRange(1),
    default=500,
    show_default=True,
    help='Round interval at which beyond-accuracy metrics (Gini, coverage, ARP) are recorded over time.'
)
def simulate(
    save_name: str,
    model_name: str,
    hadamard: bool,
    watch_threshold: float | None,
    epsilon: tuple[float, ...],
    alpha: tuple[float, ...],
    ts_v: tuple[float, ...],
    lmbda: float,
    rounds: int,
    seed_count: int,
    seed: int,
    seed_index: int | None,
    longtail_percentile: float,
    metric_interval: int,
) -> None:
    """
    Run bandit simulations using the KuaiRec-Small matrix.

    Policies simulated: Greedy (dot-product baseline), Random, ε-greedy, LinUCB, and Thompson Sampling.
    """

    flags = locals()

    # Validate --seed-index is within range if provided
    if seed_index is not None and seed_index >= seed_count:
        raise click.BadParameter(
            f'--seed-index={seed_index} is out of range for --seed-count={seed_count} (must be 0 to {seed_count - 1}).',
            param_hint='--seed-index'
        )

    # Set up output directory
    path = DLB_DIR + f'simulations/{save_name}/'
    if seed_index is not None:
        # Slurm per-seed run: don't wipe existing directory (other seeds may already be there)
        os.makedirs(path, exist_ok=True)
        if not os.path.exists(path + 'flags.json'):
            with open(path + 'flags.json', 'w') as f:
                json.dump(flags, f, indent=4)
    else:
        if os.path.exists(path): shutil.rmtree(path)
        os.makedirs(path)
        with open(path + 'flags.json', 'w') as f:
            json.dump(flags, f, indent=4)

    # Show user flags
    print(f"\nsimulate flags (logged to {path}flags.json):")
    for flag_name, flag_arg in flags.items():
        print(f"    {flag_name}: {str(flag_arg)}")

    # Load two-tower model
    model_path = DLB_DIR + f'tt-models/{model_name}/'
    model_pt = model_path + 'model.pt'
    if not os.path.exists(model_pt):
        raise click.ClickException(f"No model weights found at {model_pt}.")

    with open(model_path + 'model_args.pkl', 'rb') as f:
        model_args = pickle.load(f)
    model = dlb_tt.TwoTower(**model_args).to(device)
    model.load_state_dict(
        torch.load(
            model_pt, 
            map_location=device, 
            weights_only=True
        )
    )
    model.eval()
    print(f"\nLoaded two-tower model from {model_path}")

    # Read watch_threshold from model's flags.json if not explicitly overridden
    if watch_threshold is None:
        with open(model_path + 'flags.json', 'r') as f:
            model_flags = json.load(f)
        watch_threshold = model_flags['watch_threshold']
        print(f"Using watch_threshold={watch_threshold} from model's flags.json")

    print(f"\nLoading KuaiRec-Small (watch_threshold={watch_threshold})...")
    small_matrix = dlb_data.KRSmall(DATA_DIR, watch_threshold)
    item_popularity = dlb_data.compute_item_popularity(DATA_DIR, small_matrix.unique_item_ids, watch_threshold)

    # Build context vectors in inference mode (much faster, gradient calculations not needed)
    print(f"Building context vectors (hadamard={hadamard})...")
    with torch.inference_mode():
        user_embeddings = model.user_tower(*small_matrix.tower_ready_users(device))
        item_embeddings = model.item_tower(*small_matrix.tower_ready_items(device))
        contexts = dlb_sim.build_two_tower_contexts(user_embeddings, item_embeddings, include_product=hadamard)

    print(f"Context shape: {tuple(contexts.shape)} (D={contexts.shape[-1]})")

    # Convert to numpy for CPU-based simulation (avoids GPU overhead in per-round loop)
    # The work is mostly CPU-bound anyway & Batch Compute System GPU nodes are scarce
    contexts_np = contexts.cpu().numpy()
    user_embeddings_np = user_embeddings.cpu().numpy()
    item_embeddings_np = item_embeddings.cpu().numpy()

    # Create simulator & run
    print("\nRunning the simulator...")
    simulator = dlb_sim.Simulator(small_matrix, contexts_np, user_embeddings_np, item_embeddings_np)
    results = simulator.run(
        seed_count=seed_count,
        rounds=rounds,
        e_greedy_epsilons=list(epsilon),
        linucb_alphas=list(alpha),
        ts_vs=list(ts_v),
        lam=lmbda,
        seed=seed,
        seed_index=seed_index
    )
    print("Simulator complete!")

    labels = results['labels']

    # If a specific seed was run only (i.e. spread across multiple jobs, this is just one of them; dispatched via Slurm)
    if seed_index is not None:
        # Save only this seed's data, ready for collating
        npz_path = path + f'seed_{seed_index}.npz'
        np.savez(
            npz_path,
            rewards=results['rewards'],
            regrets=results['regrets'],
            recommendations=results['recommendations'],
            item_popularity=item_popularity
        )
        print(f"\nSeed {seed_index} results saved to {npz_path}")
        print(f"Run 'dlb collate --save-name {save_name}' after all seeds complete to generate plots & metrics.")

    # Not split into multiple Slurm jobs; either one seed only or multiple seeds within this single program instance
    else:
        # Compute all metrics (reward/regret curves + diversity metrics) across seeds
        metrics, raw_arrays = dlb_sim.compute_all_metrics(
            results["all_rewards"],
            results["all_regrets"],
            results["all_recommendations"],
            item_popularity,
            longtail_percentile,
            metric_interval
        )

        # Add data necessary for plotting & general reproducibility
        metrics_for_plot = {
            'labels':              labels,
            'seed':                seed,
            'seed_count':          seed_count,
            'longtail_percentile': longtail_percentile,
            **metrics,
        }

        # Generate all plots, using `plot.py` now instead for separation of concerns
        # Note that this means that I can use the BCS to retrieve metrics and alter plots etc still
        # Plus just makes the CLI easier to use
        dlb_plot.generate_all_plots(metrics_for_plot, flags, path)
        print(f"Plots saved to {path}")

        with open(path + 'metrics.json', 'w') as f:
            json.dump(metrics_for_plot, f, indent=4)

        # For efficient access of all raw data just in case needed again (if BCS busy etc)
        np.savez(
            path + 'raw_results.npz',
            all_rewards=results["all_rewards"],
            all_regrets=results["all_regrets"],
            all_recommendations=results["all_recommendations"],
            item_popularity=item_popularity,
            **raw_arrays,
        )
        print(f"Simulation complete. Results saved to {path}")

@cli.command('collate')
@click.option(
    '--save-name',
    type=str,
    required=True,
    help='Directory name under simulations/ containing per-seed .npz files to collate.'
)
def collate(
    save_name: str
) -> None:
    """
    Collate per-seed simulation results within a single directory into merged plots & metrics.

    Intended for use after running separate Slurm jobs with 'dlb simulate --seed-index N'; expects all seed_*.npz files to be present.
    """

    # Check folder exists
    path = DLB_DIR + f'simulations/{save_name}/'
    if not os.path.isdir(path):
        raise click.ClickException(f"Directory not found: {path}")

    # Load flags first so we can validate expected per-seed outputs directly
    flags_path = path + 'flags.json'
    if not os.path.exists(flags_path):
        raise click.ClickException(f"No flags.json found in {path}; cannot reconstruct policy labels.")
    with open(flags_path, 'r') as f:
        flags = json.load(f)

    # Validate seed files by count: require exactly one per expected seed
    seed_count = int(flags['seed_count'])
    seed_files = sorted(glob.glob(path + 'seed_*.npz'))
    if len(seed_files) != seed_count:
        raise click.UsageError(
            f"Cannot collate: expected {seed_count} seed_*.npz files in {path}, found {len(seed_files)}."
        )
    print(f"\nCollating {seed_count} seed files from {path}...")

    longtail_percentile = float(flags.get('longtail_percentile', 80.0)) # Just in case read from JSON as integer

    # Build the policy labels
    labels = dlb_sim.build_policy_labels(flags['epsilon'], flags['alpha'], flags['ts_v'])

    # Collect simulation arrays
    all_rewards         = []
    all_regrets         = []
    all_recommendations = []
    item_popularity     = None
    for seed_file in seed_files:
        with np.load(seed_file) as npz:
            # Each has dims (n_policies, rounds)
            all_rewards.append(npz['rewards'])
            all_regrets.append(npz['regrets'])
            all_recommendations.append(npz['recommendations'])
            if item_popularity is None:
                item_popularity = npz['item_popularity']

    # Stack to produce arrays with dims (total_seeds, n_policies, rounds) ready for computing means
    all_rewards = np.stack(all_rewards, axis=0)
    all_regrets = np.stack(all_regrets, axis=0)
    all_recommendations = np.stack(all_recommendations, axis=0)

    total_seeds = all_rewards.shape[0]

    metric_interval = int(flags.get('metric_interval', 500))

    # Compute all metrics (reward/regret curves + diversity over time) across seeds
    metrics, raw_arrays = dlb_sim.compute_all_metrics(
        all_rewards,
        all_regrets,
        all_recommendations,
        item_popularity,
        longtail_percentile,
        metric_interval
    )

    # Add data for plotting & reproducibility
    metrics_for_plot = {
        'labels':              labels,
        'seed':                flags['seed'],
        'seed_count':          total_seeds,
        'longtail_percentile': longtail_percentile,
        **metrics,
    }

    dlb_plot.generate_all_plots(metrics_for_plot, flags, path)
    print(f"Plots saved to {path}")

    with open(path + 'metrics.json', 'w') as f:
        json.dump(metrics_for_plot, f, indent=4)

    # Save combined raw arrays including per-seed diversity time-series
    np.savez(
        path + 'raw_results.npz',
        all_rewards=all_rewards,
        all_regrets=all_regrets,
        all_recommendations=all_recommendations,
        item_popularity=item_popularity,
        **raw_arrays,
    )

    # Combined arrays are persisted; clean up per-seed intermediate files
    for seed_file in seed_files:
        os.remove(seed_file)

    print(f"\nCollation complete: {total_seeds} seeds merged.")
    print(f"Removed {len(seed_files)} per-seed files from {path}")
    print(f"Results saved to {path}")

@cli.command('plot')
@click.option(
    '--save-name',
    type=str,
    required=True,
    help='Directory name under simulations/ containing metrics.json to plot from.'
)
@click.option(
    '--metric-interval',
    type=click.IntRange(1),
    default=None,
    help='Override the round interval for beyond-accuracy metrics. If omitted, uses the value from flags.json (or 500).'
)
def plot(
    save_name: str,
    metric_interval: int | None,
) -> None:
    """
    Regenerate all plots from saved simulation results without re-running the simulation.

    Reads metrics.json and flags.json from the given directory by default.
    Only loads raw_results.npz when --metric-interval is overridden (to recompute beyond-accuracy metrics at a different interval).
    Updates metrics.json to match the metric_interval seen in the new plots.
    """

    # Check for simulation path
    path = DLB_DIR + f'simulations/{save_name}/'
    if not os.path.isdir(path):
        raise click.ClickException(f"Directory not found: {path}")

    # Check for flags & read in
    flags_path = path + 'flags.json'
    if not os.path.exists(flags_path):
        raise click.ClickException(f"No flags.json found in {path}.")
    with open(flags_path, 'r') as f:
        flags = json.load(f)

    # Check for metrics & read in
    metrics_path = path + 'metrics.json'
    if not os.path.exists(metrics_path):
        raise click.ClickException(f"No metrics.json found in {path}. Run 'dlb simulate' or 'dlb collate' first.")
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    # Load beyond-accuracy metric interval from simulation flags if not supplied (fallback of 500)
    stored_interval = int(flags.get('metric_interval', 500))
    if metric_interval is None:
        metric_interval = stored_interval

    # If the interval is overridden, recompute beyond-accuracy metrics from raw recommendations and update the metrics dict
    if metric_interval != stored_interval:
        # Look for the raw results and load them
        npz_path = path + 'raw_results.npz'
        if not os.path.exists(npz_path):
            raise click.ClickException(
                f"--metric-interval override requires raw_results.npz but none found in {path}."
            )
        
        # Get long-tail percentile for recomputing long-tail coverage
        longtail_percentile = float(flags.get('longtail_percentile', 80.0))

        # Recompute beyond-accuracy metrics
        with np.load(npz_path) as npz:
            metric_rounds, all_gini, all_coverage, all_arp = dlb_sim.compute_ba_metrics_over_time(
                npz['all_recommendations'], 
                npz['item_popularity'], 
                metric_interval, 
                longtail_percentile
            )

        # Update metrics dictionary
        final_gini     = all_gini[:, :, -1]
        final_coverage = all_coverage[:, :, -1]
        final_arp      = all_arp[:, :, -1]
        metrics.update({
            'metric_rounds':           metric_rounds.tolist(),
            'mean_gini_over_time':     all_gini.mean(axis=0).tolist(),
            'std_gini_over_time':      all_gini.std(axis=0).tolist(),
            'mean_coverage_over_time': all_coverage.mean(axis=0).tolist(),
            'std_coverage_over_time':  all_coverage.std(axis=0).tolist(),
            'mean_arp_over_time':      all_arp.mean(axis=0).tolist(),
            'std_arp_over_time':       all_arp.std(axis=0).tolist(),
            'mean_gini':               final_gini.mean(axis=0).tolist(),
            'std_gini':                final_gini.std(axis=0).tolist(),
            'mean_longtail_coverage':  final_coverage.mean(axis=0).tolist(),
            'std_longtail_coverage':   final_coverage.std(axis=0).tolist(),
            'mean_arp':                final_arp.mean(axis=0).tolist(),
            'std_arp':                 final_arp.std(axis=0).tolist(),
        })

    print(f"\nRegenerating plots from {metrics_path} (metric_interval={metric_interval})...")
    dlb_plot.generate_all_plots(metrics, flags, path)

    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Plots and (updated) metrics saved to {path}")