"""Extract MNIST images to .toml (JSON) and .png files for use with TNN."""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

from PIL import Image
import torchvision
import torchvision.transforms as transforms


def extract_mnist(
    output_dir: str,
    count: int,
    split: str = "test",
    digits: list[int] | None = None,
    data_dir: str = "/tmp/mnist_data",
):
    if digits is None:
        digits = list(range(10))

    os.makedirs(output_dir, exist_ok=True)

    dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=(split == "train"),
        download=True,
        transform=transforms.ToTensor(),
    )

    digit_counts = defaultdict(int)
    needed = set(digits)

    for img_tensor, label in dataset:
        label = int(label)
        if label not in needed:
            continue
        if digit_counts[label] >= count:
            continue

        idx = digit_counts[label]
        digit_counts[label] += 1

        base_name = f"mnist_{split}_digit_{label}_{idx:04d}"
        values = img_tensor.squeeze().flatten().tolist()
        values = [round(v, 4) for v in values]

        toml_path = os.path.join(output_dir, f"{base_name}.toml")
        with open(toml_path, "w") as f:
            json.dump({"values": values}, f, indent=2)

        png_path = os.path.join(output_dir, f"{base_name}.png")
        img_array = img_tensor.squeeze().numpy()
        img_uint8 = (img_array * 255).clip(0, 255).astype("uint8")
        Image.fromarray(img_uint8, mode="L").save(png_path)

        if all(digit_counts[d] >= count for d in needed):
            break

    for d in digits:
        print(f"  digit {d}: {digit_counts[d]} images")


def main():
    parser = argparse.ArgumentParser(
        description="Extract MNIST images to .toml and .png files for TNN."
    )
    parser.add_argument(
        "output_dir",
        help="Directory to write output files",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of images to extract per digit (default: 1)",
    )
    parser.add_argument(
        "--split",
        choices=["train", "test"],
        default="test",
        help="Dataset split to use (default: test)",
    )
    parser.add_argument(
        "--digits",
        type=int,
        nargs="+",
        metavar="DIGIT",
        help="Digits to extract (default: all 0-9)",
    )
    parser.add_argument(
        "--data-dir",
        default="/tmp/mnist_data",
        help="Directory to cache the downloaded MNIST data (default: /tmp/mnist_data)",
    )
    args = parser.parse_args()

    digits = args.digits if args.digits else list(range(10))
    invalid = [d for d in digits if not 0 <= d <= 9]
    if invalid:
        parser.error(f"Invalid digits: {invalid}")

    print(f"Extracting {args.count} image(s) per digit from MNIST {args.split} set")
    print(f"Digits: {digits}")
    print(f"Output: {args.output_dir}")

    extract_mnist(
        output_dir=args.output_dir,
        count=args.count,
        split=args.split,
        digits=digits,
        data_dir=args.data_dir,
    )

    print("Done.")


if __name__ == "__main__":
    main()
