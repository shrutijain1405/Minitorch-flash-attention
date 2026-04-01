# import os
# import time
# import random
# import numpy as np
# from functools import partial
# import tqdm
# import fire
# import json

# import datasets
# # from torchvision import transforms
# # from PIL import Image
# import sys
# sys.path.append("./")
# import minitorch
# from minitorch import TensorBackend
# from minitorch.cuda_kernel_ops import CudaKernelOps

# from minitorch.transformer import ViT   # your model


# # -----------------------------
# # ImageNet dataset loader
# # -----------------------------
# def get_imagenet_dataset(model_image_size=224):
#     """
#     Loads ImageNet (or subset if local HF version)
#     """
#     from huggingface_hub import login
#     login(token="")

#     dataset = datasets.load_dataset("cifar10")

#     # IMPORTANT: Image preprocessing
#     # transform = transforms.Compose([
#     #     transforms.Resize((model_image_size, model_image_size)),
#     #     transforms.ToTensor(),
#     #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#     #                          std=[0.229, 0.224, 0.225]),
#     # ])

#     def preprocess(example):
#         image = example["image"].convert("RGB")
#         return {
#             "pixel_values": np.array(image),
#             "label": example["label"]
#         }

#     # dataset = {
#     #     split: dataset[split].map(preprocess)
#     #     for split in dataset.keys()
#     # }

#     return dataset


# # -----------------------------
# # Collate batch
# # -----------------------------
# def collate_batch(examples, backend):
#     images = np.stack([e["pixel_values"] for e in examples])
#     labels = np.array([e["label"] for e in examples])

#     images = minitorch.tensor_from_numpy(images, backend=backend)
#     labels = minitorch.tensor_from_numpy(labels, backend=backend)

#     return {
#         "images": images,
#         "labels": labels
#     }


# # -----------------------------
# # Loss (classification)
# # -----------------------------
# def loss_fn(batch, model):
#     logits = model(batch["images"])  # (B, 1000)

#     bs, n_classes = logits.shape

#     logits = logits.view(bs * n_classes, )
#     targets = batch["labels"]

#     loss = minitorch.nn.softmax_loss(
#         logits=logits.view(bs, n_classes),
#         target=targets
#     )

#     return loss


# # -----------------------------
# # Accuracy
# # -----------------------------
# def accuracy(logits, labels):
#     preds = np.argmax(logits.to_numpy(), axis=1)
#     return (preds == labels.to_numpy()).mean()


# # -----------------------------
# # Train loop
# # -----------------------------
# def train(model, optimizer, examples, batch_size, backend):
#     model.train()
#     random.shuffle(examples)

#     for i in tqdm.trange(0, len(examples), batch_size):
#         batch = collate_batch(examples[i:i+batch_size], backend)

#         optimizer.zero_grad()
#         loss = loss_fn(batch, model)
#         loss.backward()
#         optimizer.step()


# # -----------------------------
# # Eval
# # -----------------------------
# def evaluate(model, examples, batch_size, backend):
#     model.eval()
#     losses = []
#     accs = []

#     for i in tqdm.trange(0, len(examples), batch_size):
#         batch = collate_batch(examples[i:i+batch_size], backend)

#         logits = model(batch["images"])
#         loss = loss_fn(batch, model)

#         losses.append(loss.item())
#         accs.append(accuracy(logits, batch["labels"]))

#     return np.mean(losses), np.mean(accs)


# # -----------------------------
# # MAIN
# # -----------------------------
# def main(
#     batch_size=64,
#     lr=3e-4,
#     epochs=10,
#     n_classes=1000,
#     image_size=224
# ):

#     backend = minitorch.TensorBackend(CudaKernelOps)

#     # ---------------- ViT-Base ----------------
#     model = ViT(
#         n_embd=768,
#         n_head=12,
#         n_channels=3,
#         patch_size=16,
#         n_trans_layers=12,
#         n_classes=n_classes,
#         backend=backend
#     )

#     optimizer = minitorch.Adam(model.parameters(), lr=lr)

#     dataset = get_imagenet_dataset(image_size)

#     for epoch in range(epochs):
#         print(f"\nEpoch {epoch}")

#         train(
#             model,
#             optimizer,
#             dataset["train"],
#             batch_size,
#             backend
#         )

#         val_loss, val_acc = evaluate(
#             model,
#             dataset["validation"],
#             batch_size,
#             backend
#         )

#         print(f"Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}")

#         json.dump(
#             {"val_loss": float(val_loss), "val_acc": float(val_acc)},
#             open(f"vit_epoch_{epoch}.json", "w")
#         )


# if __name__ == "__main__":
#     fire.Fire(main)


# import torch
# import torch.nn as nn
import sys
sys.path.append("./")
import minitorch
from minitorch import TensorBackend
from minitorch.cuda_kernel_ops import CudaKernelOps

from minitorch.transformer import ViT   # your mode
from minitorch.tensor_functions import (zeros, ones, rand, tensor)

model = ViT(
        n_embd=768,
        n_head=12,
        n_channels=3,
        patch_size=16,
        n_trans_layers=12,
        n_classes=10,
        backend=minitorch.TensorBackend(CudaKernelOps)
    )
model.train()

learning_rate = 1e-4

optimizer = minitorch.Adam(model.parameters(), lr=learning_rate)

batch_size = 1
num_classes = 10

x = minitorch.rand((batch_size, 3, 224, 224), backend=minitorch.TensorBackend(CudaKernelOps))

y = minitorch.tensor([3], backend=minitorch.TensorBackend(CudaKernelOps))

logits = model(x) 

loss = minitorch.nn.softmax_loss(
        logits=logits,
        target=y
    )

print("Loss:", loss.item())

# ---- backward ----
optimizer.zero_grad()
loss.backward()
optimizer.step()
