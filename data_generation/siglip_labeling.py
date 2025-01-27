import argparse
import glob
import os
import sys

import torch
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from datasets import Dataset
from transformers import CLIPProcessor, CLIPModel


def parse_args():
  parser = argparse.ArgumentParser(
    description="Assign multi-label categories using CLIP."
  )
  parser.add_argument(
    "--data_path",
    type=str,
    default="../data",
    help="Base path containing the label folders.",
  )
  parser.add_argument(
    "--folders",
    nargs="+",
    default=["Luxurious", "Cozy", "Romantic", "Minimalist", "Scandinavian", "Vintage"],
    help="List of folder names to process (each folder is an original label).",
  )
  parser.add_argument(
    "--model_name",
    type=str,
    default="openai/clip-vit-base-patch32",
    help="Hugging Face model name for the CLIP model.",
  )
  parser.add_argument(
    "--threshold",
    type=float,
    default=0.1,
    help="Probability threshold above which a label is assigned.",
  )
  parser.add_argument(
    "--save_to_disk",
    type=str,
    default=None,
    help="Optional path to save the resulting HF dataset (arrow format).",
  )
  return parser.parse_args()


def main():
  args = parse_args()

  device = "cuda" if torch.cuda.is_available() else "cpu"
  print(f"Using device: {device}", file=sys.stderr)

  print(f"Loading model '{args.model_name}'...", file=sys.stderr)
  model = CLIPModel.from_pretrained(args.model_name).to(device)
  processor = CLIPProcessor.from_pretrained(args.model_name)

  label_prompts = {
    "Luxurious": [
      "luxurious bedroom",
      "luxurious interior",
      "opulent living room",
      "luxury hotel room",
    ],
    "Cozy": [
      "cozy bedroom",
      "warm and inviting interior",
      "cozy living room",
      "comfortable home decor",
    ],
    "Romantic": ["romantic bedroom", "romantic setting", "love-themed interior"],
    "Minimalist": [
      "minimalist interior",
      "simple white decor",
      "clean lines modern home",
    ],
    "Scandinavian": [
      "scandinavian interior",
      "nordic style decor",
      "light wood minimalist design",
    ],
    "Vintage": [
      "vintage interior",
      "retro style decor",
      "old-fashioned antique design",
    ],
  }

  threshold = args.threshold

  def get_additional_labels(image: Image.Image) -> list:
    """
    Returns a list of label strings that pass the probability threshold.
    """
    text_list = []
    label_index_map = []

    for label_name, prompts in label_prompts.items():
      for prompt in prompts:
        text_list.append(prompt)
        label_index_map.append(label_name)

    inputs = processor(
      text=text_list, images=image, return_tensors="pt", padding=True
    ).to(device)

    with torch.no_grad():
      outputs = model(**inputs)

    logits_per_image = outputs.logits_per_image  # [1, num_text_prompts]
    probs = (
      torch.sigmoid(logits_per_image).cpu().numpy().flatten()
    )  # [num_text_prompts]

    label_max_probs = {}
    for label_name, prob in zip(label_index_map, probs):
      if label_name not in label_max_probs:
        label_max_probs[label_name] = prob
      else:
        label_max_probs[label_name] = max(label_max_probs[label_name], prob)

    assigned_labels = []
    for label_name, max_prob in label_max_probs.items():
      if max_prob >= threshold:
        assigned_labels.append(label_name)

    return assigned_labels

  all_images = []
  all_original_labels = []
  all_label_lists = []

  for folder_name in args.folders:
    folder_path = os.path.join(args.data_path, folder_name)
    image_paths = glob.glob(os.path.join(folder_path, "*.jpg"))

    if not image_paths:
      print(f"Warning: No JPG images found in {folder_path}", file=sys.stderr)

    for img_path in tqdm(image_paths, desc=f"Processing {folder_name}"):
      try:
        image = Image.open(img_path).convert("RGB")
      except (FileNotFoundError, UnidentifiedImageError) as e:
        print(f"Skipping {img_path}: {e}", file=sys.stderr)
        continue

      original_label = folder_name

      additional_labels = get_additional_labels(image)

      # Merge the folder label + additional labels, removing duplicates
      final_labels = list({original_label} | set(additional_labels))

      all_images.append(img_path)
      all_original_labels.append(original_label)
      all_label_lists.append(final_labels)

  print("Creating Hugging Face Dataset...", file=sys.stderr)
  hf_dataset = Dataset.from_dict(
    {
      "image": all_images,
      "original_label": all_original_labels,
      "text": all_label_lists,
    }
  )

  if args.save_to_disk:
    print(f"Saving dataset to {args.save_to_disk}...", file=sys.stderr)
    hf_dataset.save_to_disk(args.save_to_disk)
    print("Dataset saved successfully.", file=sys.stderr)
  else:
    print("No save_to_disk path specified; dataset was not saved.", file=sys.stderr)

  print("Processing completed.", file=sys.stderr)


if __name__ == "__main__":
  main()
