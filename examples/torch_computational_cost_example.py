import argparse

from detectionmetrics.models.torch import TorchImageSegmentationModel


def parse_args() -> argparse.Namespace:
    """Parse user input arguments

    :return: parsed arguments
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, required=True, help="Scripted pytorch model"
    )
    parser.add_argument(
        "--ontology",
        type=str,
        required=True,
        help="JSON file containing model output ontology",
    )
    parser.add_argument(
        "--model_cfg",
        type=str,
        required=True,
        help="JSON file withm model configuration (norm. parameters, image size, etc.)",
    )
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()

    model = TorchImageSegmentationModel(args.model, args.model_cfg, args.ontology)
    computational_cost = model.get_computational_cost()

    print("--- Computational cost ---")
    print(f"Input shape: {computational_cost['input_shape']}")
    print(f"Model size: {computational_cost['size_mb']:.2f} MB")
    print(f"Number of parameters: {computational_cost['n_params'] / 1e6:.2f} M")
    print(f"Inference time: {computational_cost['time_s'] * 1000:.2f} ms")


if __name__ == "__main__":
    main()
