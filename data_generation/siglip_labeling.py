import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import glob
from tqdm import tqdm
from datasets import Dataset

# 1. Load model and processor
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name).cuda()
processor = CLIPProcessor.from_pretrained(model_name)

# 2. Label prompts
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
  "Vintage": ["vintage interior", "retro style decor", "old-fashioned antique design"],
}

threshold = 0.1


def get_labels_for_image(image):
  text_list = []
  label_index_map = []
  for label_name, prompts in label_prompts.items():
    for prompt in prompts:
      text_list.append(prompt)
      label_index_map.append(label_name)

  inputs = processor(
    text=text_list, images=image, return_tensors="pt", padding=True
  ).to("cuda")

  with torch.no_grad():
    outputs = model(**inputs)
  logits_per_image = outputs.logits_per_image
  probs = torch.sigmoid(logits_per_image).cpu().numpy().flatten()

  # find max prob for each label
  label_max_probs = {}
  for label_name, prob in zip(label_index_map, probs):
    label_max_probs[label_name] = max(prob, label_max_probs.get(label_name, 0))

  assigned_labels = [lbl for lbl, p in label_max_probs.items() if p >= threshold]
  if not assigned_labels:
    assigned_labels = ["Other"]
  return assigned_labels


# 3. Scan folders
folders = [
  "Luxurious",
  "Cozy",
  "Romantic",
  "Minimalist",
  "Scandinavian",
  "Vintage",
  "Other",
]
base_path = "../data"

all_images = []
all_labels = []

for folder_name in folders:
  folder_path = f"{base_path}/{folder_name}"
  image_paths = glob.glob(folder_path + "/*.jpg")
  for img_path in tqdm(image_paths, desc=f"Processing {folder_name}"):
    image = Image.open(img_path).convert("RGB")
    assigned = get_labels_for_image(image)
    all_images.append(img_path)
    all_labels.append(assigned)

# 4. Build HF dataset
hf_dataset = Dataset.from_dict({"image": all_images, "text": all_labels})

# 5. (Optional) Save or push to Hub
hf_dataset.save_to_disk("dataset_name")
