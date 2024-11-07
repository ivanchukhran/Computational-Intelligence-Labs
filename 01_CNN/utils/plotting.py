import torch
import numpy as np
import matplotlib.pyplot as plt

from typing import Dict, Any


def grid_images(
        images: list[np.ndarray], 
        titles: list[str]=None, 
        rows: int = None, 
        figsize: tuple = None
    ):
    if not rows:
        rows = int(np.sqrt(len(images)))
    cols = len(images) // rows
    fig, axs = plt.subplots(rows, cols, figsize=figsize)
    for i, ax in enumerate(axs.flat):
        ax.imshow(images[i])
        if titles:
            ax.set_title(titles[i])
        ax.axis("off")
    plt.show()


def show_results(
    orientation: str = "horizontal",
    accuracy_bottom: Any = None,
    loss_top: Any = None,
    **histories: Dict[str, Any],
) -> None:
    if orientation == "horizontal":
        f, ax = plt.subplots(1, 2, figsize=(16, 5))
    else:
        f, ax = plt.subplots(2, 1, figsize=(16, 16))
    for i, (name, h) in enumerate(histories.items()):
        if len(histories) == 1:
            ax[0].set_title(
                "Best validation accuracy: {:.2f}% (train: {:.2f}%)".format(
                    max(h["validation_accuracy"]) * 100, max(h["train_accuracy"]) * 100
                )
            )
        else:
            ax[0].set_title("Accuracy")
        ax[0].plot(h["train_accuracy"], color="C%s" % i, linestyle="--", label="%s train" % name)
        ax[0].plot(h["validation_accuracy"], color="C%s" % i, label="%s validation" % name)
        ax[0].set_xlabel("epochs")
        ax[0].set_ylabel("accuracy")
        if accuracy_bottom:
            ax[0].set_ylim(bottom=accuracy_bottom)
        ax[0].legend()

        if len(histories) == 1:
            ax[1].set_title(
                "Minimal train loss: {:.4f} (validation: {:.4f})".format(
                    min(h["train_loss"]), min(h["validation_loss"])
                )
            )
        else:
            ax[1].set_title("Loss")
        ax[1].plot(h["train_loss"], color="C%s" % i, linestyle="--", label="%s train" % name)
        ax[1].plot(h["validation_loss"], color="C%s" % i, label="%s validation" % name)
        ax[1].set_xlabel("epochs")
        ax[1].set_ylabel("loss")
        if loss_top:
            ax[1].set_ylim(top=loss_top)
        ax[1].legend()

    plt.show()