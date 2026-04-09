"""
Helper methods for looking at the results of all of the models
for the sake of the report & for general analysis
"""

import os
import json
import click

# Get DLB_DIR from environment, or fall back on cwd
DLB_DIR = os.environ.get("DLB_DIR", os.getcwd())
if not DLB_DIR.endswith("/"):
    DLB_DIR += "/"
TT_DIR = DLB_DIR + "tt-models/"

# Define these helpers as part of a CLI, for ease of use
@click.group()
def cli() -> None:
    if not os.path.exists(TT_DIR):
        raise Exception(f"The two-tower models are expected to have been placed within {TT_DIR}.")

@cli.command('best-tt')
@click.option(
    '--sort-by',
    type=click.Choice(('recall', 'ndcg')),
    default='recall',
    show_default=True,
    help='The metric class to use for sorting the two-tower models.'
)
@click.option(
    '--k',
    'sort_by_k',
    type=click.IntRange(1),
    default=50,
    show_default=True,
    help='The K to use for the Recall@K or NDCG@K model sorting. Note that models should all contain the metric for which you are sorting by (not an issue if using defaults).'
)
def best_tt(
    sort_by: str,
    sort_by_k: int
):
    """
    Takes all of the trained two-tower models in tt-models/ & prints out their saved performance on their different metrics, sorted by the specified metric. Note of course that for fair evaluation these should all
    be trained on the same seed and other fixed control variables as appropriate.
    """

    models = []
    skipped = 0

    for model in os.listdir(TT_DIR):
        path = TT_DIR + model
        if not os.path.isdir(path):
            continue

        # Gather what Ks this model saved Recall@K and NDCG@K for
        all_k = []
        with open(path + '/flags.json', 'r') as f:
            flags = json.load(f)
            all_k = flags['metric_k']
        if sort_by_k not in all_k:
            print(f"Model `{model}` doesn't have a `{sort_by}@{sort_by_k}`, skipping...")
            skipped += 1
            continue

        # Retrieve this model's saved metrics, figure out when it saved the model, retrieve correct results
        with open(path + '/metrics.json', 'r') as f:
            metrics = json.load(f)
            save_epoch = metrics['best_epoch'] - 1 # best_epoch is 1 ahead

            result_dict = {
                'model_name': model,
                'best_epoch': metrics['best_epoch'],
                'train_loss': metrics['train_loss'][save_epoch],
                'val_loss': metrics['val_loss'][save_epoch]
            }
            for k in all_k:
                result_dict[f'recall@{k}'] = metrics[f'recall@{k}'][save_epoch]
                result_dict[f'ndcg@{k}'] = metrics[f'ndcg@{k}'][save_epoch]

            models.append(result_dict)

    # Sort and print (with best model at bottom for readability in terminal)
    print()
    models.sort(key = lambda x : x[f'{sort_by}@{sort_by_k}'])
    for i, model in enumerate(models):
        print(f"{len(models) - i}. ")
        for k,v in model.items():
            print(f"\t{k}: {v}")
        print("")
    print(f"Above are all {TT_DIR} sorted by `{sort_by}@{sort_by_k}`, where available.")
    print(f"(any models without a `{sort_by}@{sort_by_k}` were skipped, of which there were {skipped})")