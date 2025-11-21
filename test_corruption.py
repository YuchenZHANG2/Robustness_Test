
import argparse
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

if not hasattr(np, "float_"):
    np.float_ = np.float64
if not hasattr(np, "complex_"):
    np.complex_ = np.complex128

import skimage.filters
_orig_gaussian = skimage.filters.gaussian

def _patched_gaussian(*args, **kwargs):
    if "multichannel" in kwargs:
        channel = kwargs.pop("multichannel")
        kwargs.setdefault("channel_axis", -1 if channel else None)
    return _orig_gaussian(*args, **kwargs)

skimage.filters.gaussian = _patched_gaussian

from imagecorruptions import corrupt, get_corruption_names


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize imagecorruptions grid")
    parser.add_argument(
        "--dataset_dir",
        default="/home/yuchen/YuchenZ/Datasets/coco/val2017",
        help="Directory containing source images",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Specific image filename to use (defaults to random .jpg in dataset_dir)",
    )
    parser.add_argument(
        "--all_images",
        action="store_true",
        help="Process all images in dataset_dir instead of just one",
    )
    parser.add_argument(
        "--subset",
        default="all",
        choices=["common", "validation", "all", "noise", "blur", "weather", "digital"],
        help="Corruption subset to render",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Call plt.show() (useful in interactive sessions)",
    )
    parser.add_argument(
        "--save",
        default="test.png",
        help="Optional path to save the figure (png)",
    )
    return parser.parse_args()


def pick_image(dataset_dir: Path, image_name: str | None) -> Path:
    if image_name:
        return dataset_dir / image_name
    choices = [p for p in dataset_dir.glob("*.jpg")]
    if not choices:
        raise FileNotFoundError(f"No .jpg files in {dataset_dir}")
    return random.choice(choices)


def main():
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).expanduser()
    
    if args.all_images:
        img_paths = sorted(dataset_dir.glob("*.jpg"))
        if not img_paths:
            raise FileNotFoundError(f"No .jpg files in {dataset_dir}")
        print(f"Processing {len(img_paths)} images from {dataset_dir}")
    else:
        img_paths = [pick_image(dataset_dir, args.image)]
        print(f"Using image: {img_paths[0]}")

    corruptions = get_corruption_names(args.subset)
    severities = [1, 2, 3, 4, 5]

    for img_path in img_paths:
        print(f"Processing: {img_path.name}")
        image = np.array(Image.open(img_path).convert("RGB"))

        rows = len(corruptions)
        cols = len(severities) + 1  # +1 for clean image column
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2.2))

        # Add column titles
        for c in range(cols):
            if c == 0:
                axes[0, c].set_title("Clean", fontsize=12, fontweight='bold')
            else:
                axes[0, c].set_title(f"Severity {severities[c-1]}", fontsize=12, fontweight='bold')

        for r, corruption in enumerate(corruptions):
            # First column: clean image
            ax = axes[r, 0]
            ax.imshow(image)
            ax.text(
                -0.25,
                0.5,
                corruption.replace("_", "\n"),
                fontsize=9,
                transform=ax.transAxes,
                ha="right",
                va="center",
            )
            ax.axis("off")
            
            # Remaining columns: corrupted images at different severities
            for c, severity in enumerate(severities, start=1):
                ax = axes[r, c]
                corrupted = corrupt(image, corruption_name=corruption, severity=severity)
                ax.imshow(corrupted)
                ax.axis("off")

        fig.suptitle(f"{img_path.name} – {args.subset} corruptions", fontsize=16)
        plt.tight_layout()

        if args.all_images:
            # When processing all images, just close the figure to free memory
            plt.close(fig)
        else:
            # When processing single image, save and/or show as requested
            if args.save:
                out_path = Path(args.save)
                fig.savefig(out_path, dpi=150)
                print(f"Saved figure to {out_path}")

            if args.show:
                plt.show()
            else:
                plt.close(fig)


if __name__ == "__main__":
    main()
