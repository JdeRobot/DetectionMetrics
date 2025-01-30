import argparse
import json

from detectionmetrics.datasets import GaiaImageSegmentationDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="GAIA-formatted image dataset Parquet file name",
    )
    parser.add_argument(
        "--new_ontology",
        type=str,
        required=True,
        help="New ontology JSON file name",
    )
    parser.add_argument(
        "--ontology_translation",
        type=str,
        required=True,
        help="Ontology translation JSON file name",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="Directory to save the new dataset",
    )
    return parser.parse_args()


def main():
    """Change the ontology of a GAIA-formatted dataset."""
    args = parse_args()

    dataset = GaiaImageSegmentationDataset(args.dataset)

    with open(args.new_ontology, "r", encoding="utf-8") as f:
        new_ontology = json.load(f)
    with open(args.ontology_translation, "r", encoding="utf-8") as f:
        ontology_translation = json.load(f)

    dataset.export(args.outdir, new_ontology, ontology_translation)


if __name__ == "__main__":
    main()
