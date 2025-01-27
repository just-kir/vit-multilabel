import torch
import numpy as np
from PIL import Image
from transformers import AutoModelForImageClassification, AutoImageProcessor


def export_vit_to_onnx(
  model_dir: str, onnx_export_path: str = "vit_model.onnx", opset_version: int = 12
):
  """
  Exports a fine-tuned Vision  model to ONNX format,
  """

  print(f"Loading model from {model_dir}...")
  model = AutoModelForImageClassification.from_pretrained(model_dir)
  processor = AutoImageProcessor.from_pretrained(model_dir)
  model.eval()

  # dummy image (shape [224, 224, 3]) as a PIL Image
  dummy_image_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
  dummy_image = Image.fromarray(dummy_image_array)

  inputs = processor(images=dummy_image, return_tensors="pt")
  pixel_values = inputs["pixel_values"]

  print(f"Exporting model to {onnx_export_path}...")
  torch.onnx.export(
    model,
    pixel_values,
    onnx_export_path,
    input_names=["pixel_values"],
    output_names=["logits"],
    dynamic_axes={"pixel_values": {0: "batch_size"}, "logits": {0: "batch_size"}},
    opset_version=opset_version,
  )

  print(f"Model successfully exported to {onnx_export_path}.")


if __name__ == "__main__":
  export_vit_to_onnx(
    model_dir="./output", onnx_export_path="vit.onnx", opset_version=12
  )
