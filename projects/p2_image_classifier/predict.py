#!/usr/bin/env python3

import argparse
import json
import os
import sys

from predict_utils import predict, load_model


def exit_if_not_file(filename):
    if not os.path.isfile(filename):
        print(f"File {filename} not found, aborting.")
        sys.exit(1)


def create_argument_parser():
    parser = argparse.ArgumentParser(
        description="Predict the top flower names from an image along with their corresponding probabilities."
    )
    parser.add_argument(
        "--top_k",
        metavar="K",
        type=int,
        default=5,
        help="Return the top K most likely classes. Defaults to 5.",
    )
    parser.add_argument(
        "--category_names",
        metavar="JSON_FILE",
        type=str,
        help="Path to a JSON file mapping labels to flower names",
    )

    parser.add_argument(
        "image", help="Path to the image of the flower that shall be predicted.",
    )

    parser.add_argument(
        "saved_model", help="Path to the Keras model used for prediction.",
    )

    return parser


def load_category_names(category_names_path):
    class_names = None
    if category_names_path is not None:
        exit_if_not_file(category_names_path)
        with open(category_names_path, "r") as f:
            class_names = json.load(f)
    return class_names


def print_results(image_path, probs, classes, class_names):
    print(f"\nThe top {len(probs)} predictions for {image_path} are:\n")
    for prob, label_index in zip(probs, classes):
        if class_names is not None:
            class_name = class_names[label_index]
        else:
            class_name = label_index
        print(f"{prob:>7.3%} {class_name}")


def main():
    args = create_argument_parser().parse_args()

    image_path = args.image
    model_path = args.saved_model

    exit_if_not_file(image_path)
    exit_if_not_file(model_path)

    class_names = load_category_names(args.category_names)
    model = load_model(model_path)
    probs, classes = predict(image_path, model, args.top_k)
    print_results(image_path, probs, classes, class_names)


main()
