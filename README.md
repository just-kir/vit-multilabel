
## Data generation
Fetches Unsplash API and deduplicates. 

Example usage:

uv run data_generation/fetch_unsplash.py \
  --access_key YOUR_ACCESS_KEY \
  --save_folder ./data/luxury \
  --queries "luxury apartment" "luxury bed" \
  --images_per_query 60 \
  --remove_duplicates \
  --duplicate_threshold 2


uv run data_generation/fetch_unsplash.py \
  --access_key YOUR_ACCESS_KEY \
  --save_folder ./data/luxury \
  --collection_id 90855231 \
  --total_needed_collection 300 \
  --remove_duplicates \
  --duplicate_threshold 2


uv run data_generation/siglip_labeling.py --data_path ../data \
                          --folders Luxurious Cozy Romantic Minimalist Scandinavian Vintage \
                          --model_name openai/clip-vit-base-patch32 \
                          --threshold 0.1 \
                          --save_to_disk my_interior_dataset


## Training

uv run train/train.py