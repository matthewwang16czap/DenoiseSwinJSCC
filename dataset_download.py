import fiftyone as fo
import fiftyone.zoo as foz
from PIL import Image


def verify_and_clean(dataset: fo.Dataset) -> int:
    """
    Verifies all images in the given FiftyOne dataset.
    Deletes any samples whose files fail verification.
    Returns the number of deleted (bad) samples.
    """
    bad_ids = []
    for sample in dataset:
        try:
            with Image.open(sample.filepath) as img:
                img.verify()
        except Exception as e:
            print(f"Invalid image: {sample.filepath} ({e})")
            bad_ids.append(sample.id)

    if bad_ids:
        dataset.delete_samples(bad_ids)
    return len(bad_ids)


def main():
    # 1. Load both splits
    train_ds = foz.load_zoo_dataset("coco-2017", split="train")
    val_ds = foz.load_zoo_dataset("coco-2017", split="validation")
    # train_ds = foz.load_zoo_dataset("cifar10", split="train")  # 50k
    # val_ds = foz.load_zoo_dataset("cifar10", split="test")  # 10k

    # 2. Verify & clean each
    n_bad_train = verify_and_clean(train_ds)
    print(f"Train: deleted {n_bad_train} bad images.")

    n_bad_val = verify_and_clean(val_ds)
    print(f"Validation: deleted {n_bad_val} bad images.")


if __name__ == "__main__":
    main()
