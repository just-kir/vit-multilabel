import os
import argparse
import requests
from PIL import Image
import imagehash


def create_save_folder(folder_path: str) -> None:
  """
  Creates the save folder if it doesn't exist.
  """
  if not os.path.exists(folder_path):
    os.makedirs(folder_path, exist_ok=True)


def download_image(image_url: str, save_path: str) -> None:
  """
  Downloads a single image from `image_url` and saves it to `save_path`.
  """
  try:
    response = requests.get(image_url, stream=True, timeout=10)
    response.raise_for_status()
    with open(save_path, "wb") as f:
      for chunk in response.iter_content(chunk_size=8192):
        f.write(chunk)
  except Exception as e:
    print(f"Error downloading {image_url} - {e}")


def fetch_unsplash_photos(
  query: str, access_key: str, per_page: int = 30, total_needed: int = 50
):
  """
  Fetches photos from Unsplash for a given query.
  We'll request multiple pages until we've downloaded the desired total_needed images.
  """
  photos = []
  page = 1
  while len(photos) < total_needed:
    needed = total_needed - len(photos)
    current_per_page = min(per_page, needed)

    url = "https://api.unsplash.com/search/photos"
    params = {
      "query": query,
      "page": page,
      "per_page": current_per_page,
      "client_id": access_key,
    }

    try:
      response = requests.get(url, params=params, timeout=10)
      response.raise_for_status()
      data = response.json()

      results = data.get("results", [])
      if not results:
        # No more results available
        break

      photos.extend(results)
      page += 1

    except Exception as e:
      print(f"Error fetching photos for query '{query}' on page {page}: {e}")
      break

  return photos[:total_needed]


def fetch_photos_from_collection(
  collection_id: int, access_key: str, total_needed: int = 30, per_page: int = 30
):
  """
  Fetches up to `total_needed` photos from a given Unsplash collection,
  using pagination until we reach `total_needed` or run out of images.
  """
  photos = []
  page = 1

  while len(photos) < total_needed:
    needed = total_needed - len(photos)
    current_per_page = min(per_page, needed)

    url = f"https://api.unsplash.com/collections/{collection_id}/photos"
    params = {
      "page": page,
      "per_page": current_per_page,
      "client_id": access_key,
    }

    try:
      response = requests.get(url, params=params, timeout=10)
      response.raise_for_status()
      data = response.json()

      if not data:
        # End of collection
        break

      photos.extend(data)
      page += 1
    except Exception as e:
      print(f"Error fetching collection {collection_id} on page {page}: {e}")
      break

  return photos[:total_needed]


def remove_duplicates_by_hash(
  folder_path: str, hash_func=imagehash.phash, threshold: int = 0
):
  """
  Remove duplicate or near-duplicate images by comparing their perceptual hash.

  :param folder_path: The directory to clean.
  :param hash_func: The hashing function (e.g., phash, dhash, whash, etc.).
  :param threshold: Maximum Hamming distance to consider images "duplicates."
  :return: List of removed files.
  """
  seen_hashes = {}
  removed_files = []

  for root, dirs, files in os.walk(folder_path):
    for filename in files:
      file_path = os.path.join(root, filename)
      # Attempt to open file as image
      try:
        with Image.open(file_path) as img:
          # Compute perceptual hash
          img_hash = hash_func(img)
      except Exception:
        # If it's not an image or can't be opened, skip
        print(f"Skipping file (unreadable or not an image): {file_path}")
        continue

      # Check if we have a "similar enough" image
      duplicate_found = False
      for existing_hash, existing_path in seen_hashes.items():
        distance = img_hash - existing_hash
        if distance <= threshold:
          print(
            f"Duplicate found (distance={distance}):\n"
            f"  Keeping: {existing_path}\n"
            f"  Removing: {file_path}"
          )
          os.remove(file_path)
          removed_files.append(file_path)
          duplicate_found = True
          break

      if not duplicate_found:
        seen_hashes[img_hash] = file_path

  return removed_files


def main():
  parser = argparse.ArgumentParser(
    description="Fetch images from Unsplash by query or collection, save them locally, and optionally deduplicate."
  )

  parser.add_argument(
    "--access_key",
    type=str,
    required=True,
    help="Your Unsplash access key (client_id).",
  )

  parser.add_argument(
    "--save_folder", type=str, default="./data", help="Folder to save images."
  )

  # For queries
  parser.add_argument(
    "--queries",
    nargs="*",
    help="One or more queries to fetch images from Unsplash. Example: --queries 'luxury bed' 'luxury apartment'",
  )
  parser.add_argument(
    "--images_per_query",
    type=int,
    default=30,
    help="Number of images to fetch per query.",
  )

  # For collection
  parser.add_argument(
    "--collection_id", type=int, help="Unsplash collection ID to fetch images from."
  )
  parser.add_argument(
    "--total_needed_collection",
    type=int,
    default=30,
    help="Number of images to fetch from the collection.",
  )

  # Deduplication
  parser.add_argument(
    "--remove_duplicates",
    action="store_true",
    help="Remove duplicates using perceptual hash comparison.",
  )
  parser.add_argument(
    "--duplicate_threshold",
    type=int,
    default=0,
    help="Hamming distance threshold for duplicate detection (0 = exact match).",
  )

  args = parser.parse_args()

  # Create folder
  create_save_folder(args.save_folder)

  # If queries are provided, fetch images for each query
  if args.queries:
    for query in args.queries:
      print(f"Fetching images for query: '{query}'")
      photos = fetch_unsplash_photos(
        query=query,
        access_key=args.access_key,
        per_page=30,
        total_needed=args.images_per_query,
      )
      for idx, photo in enumerate(photos):
        image_url = photo["urls"]["regular"]
        # Incorporate the photo's ID to avoid collisions across queries
        image_id = photo["id"]
        filename = f"{query.replace(' ', '_')}_{image_id}_{idx}.jpg"
        file_path = os.path.join(args.save_folder, filename)

        print(f"  Downloading {filename}...")
        download_image(image_url, file_path)

  # If collection_id is provided, fetch from that collection
  if args.collection_id:
    print(f"Fetching images from Unsplash collection {args.collection_id}...")
    photos = fetch_photos_from_collection(
      collection_id=args.collection_id,
      access_key=args.access_key,
      total_needed=args.total_needed_collection,
      per_page=30,
    )
    for idx, photo in enumerate(photos):
      image_url = photo["urls"]["regular"]
      image_id = photo["id"]  # unique ID from Unsplash
      filename = f"{image_id}.jpg"
      file_path = os.path.join(args.save_folder, filename)

      print(f"  Downloading {filename}...")
      download_image(image_url, file_path)

  # Deduplicate if requested
  if args.remove_duplicates:
    print("\nRemoving duplicate images...")
    removed = remove_duplicates_by_hash(
      folder_path=args.save_folder,
      hash_func=imagehash.phash,
      threshold=args.duplicate_threshold,
    )
    print(f"Removed {len(removed)} files considered duplicates (or near-duplicates).")

  print("Done.")


if __name__ == "__main__":
  main()
