import sys
import pickle
import os
import argparse

sys.path.append("../../dev")
from utils import *
from model import *
from prefixes import *
from metrics import *
from run import *
from visualize import *

sys.path.append("../../model")
from model_params import *

sys.path.append("../../data")
from dataset_params import *
from prompt_params import *
from demo_params import *

def run_job(models: list, datasets: list, num_inputs: int, num_demos: int):
    print("Running job with the following arguments:")
    print("Models:", models)
    print("Datasets:", datasets)
    print("Number of Inputs:", num_inputs)
    print("Number of Demonstrations:", num_demos)

    demo_params = DEMO_PARAMS["permuted_incorrect_labels"]
    for model_name in models:
        model_params = MODEL_PARAMS[model_name]
        model, tokenizer = get_model_and_tokenizer(model_params["file_path"], 0)

        if num_demos == "max":
            n_demos = model_params["max_demos"]
        else:
            n_demos = int(num_demos)

        for dataset_name in datasets:
            dataset_params = DATASET_PARAMS[dataset_name]
            prompt_params = PROMPT_PARAMS[dataset_name]

            all_metrics_df = []

            prefixes = Prefixes(
                get_dataset(dataset_params),
                prompt_params[0],
                demo_params,
                model_params,
                tokenizer,
                num_inputs,
                n_demos,
            )
            metrics = run_attn(model, tokenizer, prefixes)
            metrics.to_csv(
                f"../../results/attention/{model_name}/{dataset_name}.csv", index=False
            )
            torch.cuda.empty_cache()

    return

def check_args(models: list, datasets: list, num_inputs: int, num_demos: int):
    our_datasets, our_models, our_settings = (
        list(DATASET_PARAMS.keys()),
        list(MODEL_PARAMS.keys()),
        list(DEMO_PARAMS.keys()),
    )

    for model in models:
        if model not in our_models:
            raise Exception("Model specified is not supported.")
    for dataset in datasets:
        if dataset not in our_datasets:
            raise Exception("Dataset specified is not supported.")

    if (not num_demos.isdigit() and num_demos != "max") or (
        num_demos.isdigit() and (int(num_demos) > 40 or int(num_demos) < 0)
    ):
        raise Exception(
            "Number of demonstrations must be in the range[0, 40] or use keyword 'max'."
        )
    elif num_inputs < 0:
        raise Exception("Number of inputs must be greater than 0.")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Define command line arguments.
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help="Models to run (use 'all' to run all models).",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="unnatural",
        help="Datasets to run (use 'all' to run all datasets).",
    )
    parser.add_argument("--num_inputs", type=int, default=250, help="Number of inputs.")
    parser.add_argument(
        "--num_demos", type=str, default="max", help="Number of demonstrations."
    )

    args = parser.parse_args()

    # Check for "all" option and updating arguments accordingly.
    models = (
        list(MODEL_PARAMS.keys()) if args.models == "all" else args.models.split(",")
    )
    datasets = (
        list(DATASET_PARAMS.keys())
        if args.datasets == "all"
        else args.datasets.split(",")
    )

    check_args(models, datasets, args.num_inputs, args.num_demos)

    # Launch the job.
    run_job(models, datasets, args.num_inputs, args.num_demos)