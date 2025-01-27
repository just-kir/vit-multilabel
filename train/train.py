"""
Script for multi-label image classification via HF Transformers.

Usage:
  1. Set the DATA_PATH environment variable to the path or identifier of your dataset.
  2. Run this script.
  3. The trained model and logs will be saved to the output directory.
"""

import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from datasets import load_dataset
from transformers import (
  AutoImageProcessor,
  AutoModelForImageClassification,
  TrainingArguments,
  Trainer,
)
from sklearn.metrics import precision_recall_fscore_support, average_precision_score


def main():
  """
  Main entry point for training a multi-label image classification model.
  """

  # --------------------------------------------------------------------
  # 1. Load Dataset
  # --------------------------------------------------------------------
  data_path = os.getenv("DATA_PATH")
  if not data_path:
    raise ValueError(
      "DATA_PATH environment variable must be set to a valid dataset path or identifier."
    )

  print(f"Loading dataset from: {data_path}")
  dataset = load_dataset(data_path)
  dataset_train = dataset["train"]

  # --------------------------------------------------------------------
  # 2. Build the label set from the train set
  # --------------------------------------------------------------------
  print("Building unique label set.")
  unique_labels = list(
    set(item for sublist in dataset_train["text"] for item in sublist)
  )
  unique_labels.sort()

  label2id = {label: i for i, label in enumerate(unique_labels)}
  id2label = {i: label for label, i in label2id.items()}
  num_classes = len(unique_labels)

  # --------------------------------------------------------------------
  # 3. Convert Multi-Label Texts to Multi-Hot Label Vectors
  # --------------------------------------------------------------------
  def encode_labels(example):
    """
    Convert a list of text labels into a multi-hot vector.
    """
    multi_hot = [0] * num_classes
    for lbl in example["text"]:
      idx = label2id[lbl]
      multi_hot[idx] = 1
    example["labels"] = multi_hot
    return example

  print("Encoding labels as multi-hot vectors.")
  dataset_train = dataset_train.map(encode_labels)

  # --------------------------------------------------------------------
  # 4. Train/Test Split
  # --------------------------------------------------------------------
  print("Performing train/test split.")
  split_dataset = dataset_train.train_test_split(test_size=0.2, seed=42)
  train_ds = split_dataset["train"]
  eval_ds = split_dataset["test"]

  # --------------------------------------------------------------------
  # 5. Set Up Transforms
  # --------------------------------------------------------------------
  print("Defining data transformations.")
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
  # 6. Load Image Processor & Model for Multi-Label
  # --------------------------------------------------------------------
  model_name = "google/vit-base-patch16-224-in21k"
  print(f"Loading image processor and model: {model_name}")
  image_processor = AutoImageProcessor.from_pretrained(model_name)

  model = AutoModelForImageClassification.from_pretrained(
    model_name,
    num_labels=num_classes,
    label2id=label2id,
    id2label=id2label,
    problem_type="multi_label_classification",
  )

  # --------------------------------------------------------------------
  # 7. Custom Data Collator (applies transforms and returns float labels)
  # --------------------------------------------------------------------
  class DataCollatorWithTransforms:
    """
    Custom data collator that applies data augmentation/transforms
    and processes images into tensors with corresponding float labels.
    """

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
        if not isinstance(aug_img, Image.Image):
          raise ValueError("Expected a PIL image after transform.")

        encoding = self.processor(aug_img, return_tensors="pt")
        pixel_values.append(encoding["pixel_values"][0])

        # Convert multi-hot label list -> float tensor
        labels.append(example["labels"])

      pixel_values = torch.stack(pixel_values)
      labels_tensor = torch.tensor(labels, dtype=torch.float)

      return {"pixel_values": pixel_values, "labels": labels_tensor}

  train_collator = DataCollatorWithTransforms(
    transform=train_transform, processor=image_processor
  )

  # --------------------------------------------------------------------
  # 8. Multi-Label Metrics
  # --------------------------------------------------------------------
  def compute_metrics(eval_pred):
    """
    Compute multi-label metrics including micro/macro Precision, Recall, F1,
    and mean Average Precision (mAP).
    """
    logits, labels = eval_pred

    # Sigmoid -> probabilities
    probs = 1 / (1 + np.exp(-logits))

    # Convert probabilities to binary predictions at a threshold of 0.5
    preds = (probs >= 0.5).astype(int)

    # Micro and Macro P/R/F1
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
      labels, preds, average="micro", zero_division=0
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
      labels, preds, average="macro", zero_division=0
    )

    # Mean Average Precision (mAP)
    num_labels_local = labels.shape[1]
    ap_scores = []
    for i in range(num_labels_local):
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
  # 9. Training Arguments & Trainer
  # --------------------------------------------------------------------
  print("Setting up training arguments.")
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

  print("Initializing Trainer.")
  trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=train_collator,
    compute_metrics=compute_metrics,
  )

  # --------------------------------------------------------------------
  # 10. Train!
  # --------------------------------------------------------------------
  print("Starting training.")
  trainer.train()
  print("Training completed.")


if __name__ == "__main__":
  main()
