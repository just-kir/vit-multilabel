# Attribute Classification

A minimal pipeline for fetching images from Unsplash, labeling them with CLIP, training a classification model, and running inference.

## Data Generation

Fetch images from Unsplash by queries or collection ID:

```bash
uv run data_generation/fetch_unsplash.py \
  --access_key YOUR_ACCESS_KEY \
  --save_folder ./data/luxury \
  --queries "luxury apartment" "luxury bed" \
  --images_per_query 60 \
  --remove_duplicates \
  --duplicate_threshold 2
```

```bash
uv run data_generation/fetch_unsplash.py \
  --access_key YOUR_ACCESS_KEY \
  --save_folder ./data/luxury \
  --collection_id 90855231 \
  --total_needed_collection 300 \
  --remove_duplicates \
  --duplicate_threshold 2
```

### Additional Labels via One-Shot Siglip
```bash
uv run data_generation/siglip_labeling.py \
  --data_path ../data \
  --folders Luxurious Cozy Romantic Minimalist Scandinavian Vintage \
  --model_name openai/clip-vit-base-patch32 \
  --threshold 0.1 \
  --save_to_disk my_interior_dataset
```

## Training
```bash
uv run train/train.py
```

## Inference
Convert to ONNX:

```bash
uv run train/train.py
```

Modal inference serving run: 

```bash
modal run inference/modal.py
```
