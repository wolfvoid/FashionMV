"""Extract FashionGen images from h5 files.

The h5 file stores images row by row. Each row has an `input_productID` field.
Rows belonging to the same product are stored consecutively. This script groups
rows by product ID and saves each view as `{h5_row_index}.jpg` under
`{output_dir}/{product_id}/`.

Usage:
  python prepare_fashiongen.py \
    --h5_path /path/to/fashiongen_256_256_validation.h5 \
    --output_dir /path/to/images/fashiongen_val

  python prepare_fashiongen.py \
    --h5_path /path/to/fashiongen_256_256_train.h5 \
    --output_dir /path/to/images/fashiongen_train
"""
import argparse
import os
from collections import OrderedDict

import h5py
from PIL import Image
from tqdm import tqdm


def build_product_mapping(h5_file):
    """Build product_id -> list of h5 row indices from the h5 file directly."""
    pids = h5_file["input_productID"]
    n = pids.shape[0]
    mapping = OrderedDict()
    for i in range(n):
        pid = str(int(pids[i][0]))
        if pid not in mapping:
            mapping[pid] = []
        mapping[pid].append(i)
    return mapping


def main():
    parser = argparse.ArgumentParser(description="Extract FashionGen images from h5")
    parser.add_argument("--h5_path", type=str, required=True, help="Path to FashionGen h5 file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output image directory")
    args = parser.parse_args()

    print(f"Loading h5: {args.h5_path}")
    h5 = h5py.File(args.h5_path, "r")

    print("Building product mapping from input_productID...")
    mapping = build_product_mapping(h5)
    total_images = sum(len(v) for v in mapping.values())
    print(f"  {len(mapping)} products, {total_images} images")

    os.makedirs(args.output_dir, exist_ok=True)
    images = h5["input_image"]

    with tqdm(total=total_images, desc="Extracting") as pbar:
        for product_id, row_indices in mapping.items():
            product_dir = os.path.join(args.output_dir, product_id)
            os.makedirs(product_dir, exist_ok=True)
            for idx in row_indices:
                out_path = os.path.join(product_dir, f"{idx}.jpg")
                if not os.path.exists(out_path):
                    Image.fromarray(images[idx]).save(out_path, quality=95)
                pbar.update(1)

    h5.close()
    print(f"Done! Images saved to {args.output_dir}")


if __name__ == "__main__":
    main()
