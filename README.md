
## Data generation
Fetches Unsplash API and deduplicates. 

Example usage:

python data_generation/fetch_unsplash.py \
  --access_key YOUR_ACCESS_KEY \
  --save_folder ./data/luxury \
  --queries "luxury apartment" "luxury bed" \
  --images_per_query 60 \
  --remove_duplicates \
  --duplicate_threshold 2


python data_generation/fetch_unsplash.py \
  --access_key YOUR_ACCESS_KEY \
  --save_folder ./data/luxury \
  --collection_id 90855231 \
  --total_needed_collection 300 \
  --remove_duplicates \
  --duplicate_threshold 2

## Training

python train/train.py