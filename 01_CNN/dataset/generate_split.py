import os 

def generate_splits(
    path: str,
    split_ratios: dict[str, float] = {"train": 0.8, "test": 0.1, "validation": 0.1},
):
    classes = os.listdir(path)
    splits: dict[str, list[str]] = {split: [] for split in split_ratios.keys()}
    for cls in classes:
        images = os.listdir(os.path.join(path, cls))
        for split, ratio in split_ratios.items():
            n = int(len(images) * ratio)
            splits[split] += [os.path.join(cls, image) for image in images[:n]]
            images = images[n:]
    return splits
