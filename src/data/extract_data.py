from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from PIL import Image


# Change this to the folder containing:
# data_batch_1, ..., data_batch_5, test_batch, batches.meta
CIFAR10_FOLDER = Path(r"/home/b629/Project/own_projects/cifar_10/cifar-10-batches-py")

# Folder where individual PNG images will be stored
OUTPUT_FOLDER = Path(r"/home/b629/Project/own_projects/cifar_10/cifar_10_images")


def load_pickle(file_path: Path) -> dict:
    """Load one CIFAR-10 pickle file."""
    with file_path.open("rb") as file:
        return pickle.load(file, encoding="bytes")


def load_class_names(dataset_folder: Path) -> list[str]:
    """Read CIFAR-10 class names from batches.meta."""
    metadata_path = dataset_folder / "batches.meta"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    metadata = load_pickle(metadata_path)

    class_names = metadata[b"label_names"]

    return [
        name.decode("utf-8") if isinstance(name, bytes) else str(name)
        for name in class_names
    ]


def extract_batch(
    batch_path: Path,
    output_folder: Path,
    class_names: list[str],
    split: str,
) -> int:
    """Extract all images from one CIFAR-10 batch."""

    if not batch_path.exists():
        raise FileNotFoundError(f"Batch file not found: {batch_path}")

    batch = load_pickle(batch_path)

    image_data = batch[b"data"]

    # Some CIFAR versions use "labels"; CIFAR-100 may use other keys.
    labels = batch.get(b"labels")

    if labels is None:
        raise KeyError(f"No labels found in {batch_path}")

    # Original shape: (N, 3072)
    # Reshape to: (N, 3, 32, 32)
    images = image_data.reshape(-1, 3, 32, 32)

    # Convert from channel-first to channel-last:
    # (N, 3, 32, 32) -> (N, 32, 32, 3)
    images = images.transpose(0, 2, 3, 1)

    for index, (image_array, label) in enumerate(zip(images, labels)):
        class_name = class_names[label]

        class_folder = output_folder / split / class_name
        class_folder.mkdir(parents=True, exist_ok=True)

        image = Image.fromarray(
            image_array.astype(np.uint8),
            mode="RGB",
        )

        # Include the source batch name so filenames remain unique.
        filename = f"{batch_path.name}_{index:05d}.png"
        image.save(class_folder / filename)

    return len(images)


def main() -> None:
    class_names = load_class_names(CIFAR10_FOLDER)

    print("Classes:", class_names)

    total_training_images = 0

    for batch_number in range(1, 6):
        batch_path = CIFAR10_FOLDER / f"data_batch_{batch_number}"

        extracted = extract_batch(
            batch_path=batch_path,
            output_folder=OUTPUT_FOLDER,
            class_names=class_names,
            split="train",
        )

        total_training_images += extracted
        print(f"Extracted {extracted} images from {batch_path.name}")

    test_batch_path = CIFAR10_FOLDER / "test_batch"

    total_test_images = extract_batch(
        batch_path=test_batch_path,
        output_folder=OUTPUT_FOLDER,
        class_names=class_names,
        split="test",
    )

    print("\nExtraction complete.")
    print(f"Training images: {total_training_images}")
    print(f"Test images: {total_test_images}")
    print(f"Saved to: {OUTPUT_FOLDER.resolve()}")


if __name__ == "__main__":
    main()