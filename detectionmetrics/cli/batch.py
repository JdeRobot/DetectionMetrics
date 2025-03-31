from itertools import product
from glob import glob
import os

import click
from tqdm import tqdm

from detectionmetrics.cli import REGISTRY as cli_registry
from detectionmetrics.utils import io as uio


@click.command(name="batch", help="Perform detection metrics jobs in batch mode")
@click.argument("command", type=click.Choice(list(cli_registry.keys())))
@click.argument("jobs_cfg", type=click.Path(dir_okay=False))
def batch(command, jobs_cfg):
    """Perform detection metrics jobs in batch mode"""
    jobs_cfg = uio.read_yaml(jobs_cfg)

    # If a single model has been provided, convert it to a list
    if not isinstance(jobs_cfg["model"], list):
        jobs_cfg["model"] = [jobs_cfg["model"]]

    # Same for dataset
    has_dataset = "dataset" in jobs_cfg
    if has_dataset and not isinstance(jobs_cfg["dataset"], list):
        jobs_cfg["dataset"] = [jobs_cfg["dataset"]]

    # Build list of model configurations
    model_cfgs = []
    for model_cfg in jobs_cfg["model"]:

        model_path = model_cfg["path"]
        model_paths = glob(model_path) if model_cfg["path_is_pattern"] else [model_path]
        assert model_paths, f"No files found for pattern {model_cfg['path']}"

        for new_path in model_paths:
            assert os.path.exists(new_path), f"File or directory {new_path} not found"

            new_path = os.path.abspath(new_path)
            new_model_id = os.path.basename(new_path)
            if os.path.isfile(new_path):
                new_model_id, _ = os.path.splitext(new_model_id)

            new_model_cfg = model_cfg | {
                "path": new_path,
                "id": f"{model_cfg['id']}-{new_model_id.replace('-', '_')}",
            }
            model_cfgs.append(new_model_cfg)

    # Build output directory and file
    os.makedirs(jobs_cfg["outdir"], exist_ok=True)
    out_fname = os.path.join(jobs_cfg["outdir"], f"results-{jobs_cfg['id']}.csv")
    if jobs_cfg.get("overwrite", False) and os.path.isfile(out_fname):
        os.remove(out_fname)

    # Build list of jobs (IDs must be unique)
    all_jobs = {}
    job_iter = product(model_cfgs, jobs_cfg["dataset"]) if has_dataset else model_cfgs
    for job_components in job_iter:
        if not isinstance(job_components, tuple):
            job_components = (job_components,)

        job_id = "-".join([str(jc["id"]) for jc in job_components])
        if job_id in all_jobs:
            raise ValueError(f"Job ID {job_id} is not unique")

        all_jobs[job_id] = job_components

    print("\n" + "-" * 80)
    print(f"{len(all_jobs)} job(s) will be executed:")
    for job_id in all_jobs:
        print(f"\t- {job_id}")
    print("-" * 80 + "\n")

    # Start processing jobs
    pbar = tqdm(all_jobs.items(), total=len(all_jobs), leave=True)
    preds_outdir = None
    for job_id, job_components in pbar:
        job_out_fname = os.path.join(jobs_cfg["outdir"], f"{job_id}.csv")
        if jobs_cfg.get("store_results_per_sample", False):
            preds_outdir = os.path.join(jobs_cfg["outdir"], f"preds-{job_id}")

        # Check if output file already exists and skip if necessary
        if os.path.isfile(job_out_fname) and not jobs_cfg.get("overwrite", False):
            pbar.write(f"Skipping {job_id}: output file already exists")
            continue

        # Execute job
        pbar.set_description(f"Processing {job_id}")

        ctx = click.get_current_context()
        try:
            params = {
                "task": jobs_cfg["task"],
                "input_type": jobs_cfg["input_type"],
            }

            model_cfg = job_components[0]
            params.update(
                {
                    "model_format": model_cfg["format"],
                    "model": model_cfg["path"],
                    "model_ontology": model_cfg["ontology"],
                    "model_cfg": model_cfg["cfg"],
                    # "image_size": model_cfg.get("image_size", None),
                }
            )
            if has_dataset:
                dataset_cfg = job_components[1]
                params.update(
                    {
                        "dataset_format": dataset_cfg.get("format", None),
                        "dataset_fname": dataset_cfg.get("fname", None),
                        "dataset_dir": dataset_cfg.get("dir", None),
                        "split_dir": dataset_cfg.get("split_dir", None),
                        "train_dataset_dir": dataset_cfg.get("train_dir", None),
                        "val_dataset_dir": dataset_cfg.get("val_dir", None),
                        "test_dataset_dir": dataset_cfg.get("test_dir", None),
                        "images_dir": dataset_cfg.get("data_dir", None),
                        "labels_dir": dataset_cfg.get("labels_dir", None),
                        "data_suffix": dataset_cfg.get("data_suffix", None),
                        "label_suffix": dataset_cfg.get("label_suffix", None),
                        "dataset_ontology": dataset_cfg.get("ontology", None),
                        "split": dataset_cfg["split"],
                        "ontology_translation": jobs_cfg.get(
                            "ontology_translation", None
                        ),
                    }
                )

            params.update({"out_fname": job_out_fname})
            if preds_outdir is not None:
                params.update({"predictions_outdir": preds_outdir})

            result = ctx.invoke(cli_registry[command], **params)

        except Exception as e:
            print(f"Error processing job {job_id}: {e}")
            continue

        # We assume that the command returns the results as a pandas DataFrame
        result["job_id"] = job_id
        result = result.set_index("job_id", append=True)

        # Write result to output file incrementally
        if not os.path.isfile(out_fname):
            result.to_csv(out_fname, mode="w", header=True)
        else:
            result.to_csv(out_fname, mode="a", header=False)


if __name__ == "__main__":
    batch()
