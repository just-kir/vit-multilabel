import random
import numpy as np
import torch
import scipy
from PIL import Image
from torchvision import transforms
import os

from datasets import load_dataset
from transformers import (
  AutoImageProcessor,
  AutoModelForImageClassification,
  TrainingArguments,
  Trainer,
)
from sklearn.metrics import precision_recall_fscore_support, average_precision_score

# --------------------------------------------------------------------
# Load Dataset
# ---------------------------------------------------------------------
dataset = load_dataset(os.getenv("DATA_PATH"))
dataset_train = dataset["train"]

# --------------------------------------------------------------------
# Build the label set from train set
# --------------------------------------------------------------------
unique_labels = list(set(item for sublist in dataset_train["text"] for item in sublist))
unique_labels.sort()

label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for label, i in label2id.items()}
num_classes = len(unique_labels)


# --------------------------------------------------------------------
# Convert Multi-Label Texts to Multi-Hot Label Vectors
# --------------------------------------------------------------------
def encode_labels(example):
  # example["text"] is a list of labels e.g. ["Modern", "Luxury", ...]
  multi_hot = [0] * num_classes
  for label in example["text"]:
    idx = label2id[label]
    multi_hot[idx] = 1
  example["labels"] = multi_hot
  return example


dataset_train = dataset_train.map(encode_labels)

# --------------------------------------------------------------------
# Train/Test Split
# --------------------------------------------------------------------
split_dataset = dataset_train.train_test_split(test_size=0.2, seed=42)
train_ds = split_dataset["train"]
eval_ds = split_dataset["test"]

# --------------------------------------------------------------------
# Set Up Transforms
# --------------------------------------------------------------------
train_transform = transforms.Compose(
  [
    transforms.RandomResizedCrop(size=224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=15),
  ]
)

eval_transform = transforms.Compose(
  [
    transforms.Resize((224, 224)),
  ]
)

# --------------------------------------------------------------------
# Load Image Processor & Model for Multi-Label
# --------------------------------------------------------------------
model_name = "google/vit-base-patch16-224-in21k"
image_processor = AutoImageProcessor.from_pretrained(model_name)

model = AutoModelForImageClassification.from_pretrained(
  model_name,
  num_labels=num_classes,
  label2id=label2id,
  id2label=id2label,
  problem_type="multi_label_classification",
)


# --------------------------------------------------------------------
# Custom Data Collator (applies transforms and returns float labels)
# --------------------------------------------------------------------
class DataCollatorWithTransforms:
  def __init__(self, transform, processor):
    self.transform = transform
    self.processor = processor

  def __call__(self, batch):
    pixel_values = []
    labels = []

    for example in batch:
      img = example["image"]
      if not isinstance(img, Image.Image):
        img = Image.fromarray(np.uint8(img))

      aug_img = self.transform(img)

      if isinstance(aug_img, Image.Image):
        encoding = self.processor(aug_img, return_tensors="pt")
      else:
        raise ValueError("Expected a PIL image after transform.")

      pixel_values.append(encoding["pixel_values"][0])

      # Convert multi-hot label list -> float tensor
      labels.append(example["labels"])

    pixel_values = torch.stack(pixel_values)
    labels = torch.tensor(labels, dtype=torch.float)  # float for BCEWithLogits

    return {"pixel_values": pixel_values, "labels": labels}


train_collator = DataCollatorWithTransforms(
  transform=train_transform, processor=image_processor
)


# --------------------------------------------------------------------
# 9. Multi-Label Metrics
# --------------------------------------------------------------------
def compute_metrics(eval_pred):
  logits, labels = eval_pred

  # Sigmoid -> probabilities
  probs = 1 / (1 + np.exp(-logits))

  # Convert probabilities to binary predictions at threshold
  preds = (probs >= 0.5).astype(int)

  # Micro and Macro P/R/F1
  precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
    labels, preds, average="micro", zero_division=0
  )
  precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
    labels, preds, average="macro", zero_division=0
  )

  # Mean Average Precision (mAP)
  num_labels = labels.shape[1]
  ap_scores = []
  for i in range(num_labels):
    ap = average_precision_score(labels[:, i], probs[:, i])
    ap_scores.append(ap)
  mAP = np.mean(ap_scores)

  return {
    "precision_micro": precision_micro,
    "recall_micro": recall_micro,
    "f1_micro": f1_micro,
    "precision_macro": precision_macro,
    "recall_macro": recall_macro,
    "f1_macro": f1_macro,
    "mAP": mAP,
  }


# --------------------------------------------------------------------
# Training Arguments & Trainer
# --------------------------------------------------------------------
training_args = TrainingArguments(
  output_dir="./output",
  per_device_train_batch_size=16,
  num_train_epochs=3,
  evaluation_strategy="steps",
  eval_steps=50,
  remove_unused_columns=False,
  logging_strategy="steps",
  logging_steps=50,
)

trainer = Trainer(
  model=model,
  args=training_args,
  train_dataset=train_ds,
  eval_dataset=eval_ds,
  data_collator=train_collator,
  compute_metrics=compute_metrics,
)

# --------------------------------------------------------------------
# Train!
# --------------------------------------------------------------------
trainer.train()
