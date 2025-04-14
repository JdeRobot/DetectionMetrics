import argparse
import json

from detectionmetrics.datasets.rellis3d import Rellis3DImageSegmentationDataset


def parse_args() -> argparse.Namespace:
    """Parse user input arguments

    :return: parsed arguments
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Directory where dataset images and labels are stored",
    )
    parser.add_argument(
        "--split_dir",
        type=str,
        required=True,
        help="Directory where .lst files defining the dataset split are stored",
    )
    parser.add_argument(
        "--ontology_fname",
        type=str,
        required=True,
        help="YAML file containing dataset ontology",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="Directory where dataset will be stored in common format",
    )
    parser.add_argument(
        "--new_ontology",
        type=str,
        help="New ontology JSON file name",
    )
    parser.add_argument(
        "--ontology_translation",
        type=str,
        help="Ontology translation JSON file name",
    )
    parser.add_argument(
        "--resize",
        type=str,
        help="Resize images to a specific size (e.g. 512x512)",
    )

    args = parser.parse_args()
    args.resize = tuple(map(int, args.resize.split("x"))) if args.resize else None
    return args


def main():
    """Main function"""
    args = parse_args()

    new_ontology, ontology_translation = None, None
    if args.new_ontology is not None:
        with open(args.new_ontology, "r", encoding="utf-8") as f:
            new_ontology = json.load(f)

    if args.ontology_translation is not None:
        with open(args.ontology_translation, "r", encoding="utf-8") as f:
            ontology_translation = json.load(f)

    dataset = Rellis3DImageSegmentationDataset(
        dataset_dir=args.dataset_dir,
        split_dir=args.split_dir,
        ontology_fname=args.ontology_fname,
    )
    dataset.export(
        args.outdir,
        resize=args.resize,
        new_ontology=new_ontology,
        ontology_translation=ontology_translation,
    )


if __name__ == "__main__":
    main()
