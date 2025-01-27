from modal import Image, App, build, enter, method, web_endpoint, Secret
import os
from fastapi import Depends, HTTPException, status, Body
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from io import BytesIO
import base64
from PIL import Image as PILImage

app = App("vit-multilabel-classifier")

auth_scheme = HTTPBearer()

image = (
  Image.debian_slim()
  .pip_install(
    "optimum[cpu]",
    "onnx",
    "onnxruntime",
    "transformers",
    "torch==2.2.0+cpu",
    "-f",
    "https://download.pytorch.org/whl/torch_stable.html",
  )
  .pip_install("Pillow")  # For image handling
)

with image.imports():
  from transformers import AutoImageProcessor
  from onnxruntime import InferenceSession
  import numpy as np

MODEL_DIR = "/"
secrets = [Secret.from_name("hf-secret"), Secret.from_name("reply-detector-auth")]

LABELS = [
  "Luxurious",
  "Cozy",
  "Romantic",
  "Minimalist",
  "Scandinavian",
  "Vintage",
  "Other",
]


@app.cls(image=image, secrets=secrets, keep_warm=1)
class Model:
  @build()
  def download_model_to_folder(self):
    from huggingface_hub import snapshot_download

    os.makedirs(MODEL_DIR, exist_ok=True)
    snapshot_download(
      "your-username/vit-multilabel-model",
      local_dir=MODEL_DIR,
      allow_patterns=[
        "*.onnx",
        "config.json",
        "preprocessor_config.json",  # if your image processor config is named differently
        # other relevant files ...
      ],
    )

  @enter()
  def setup(self):
    model_path = os.path.join(MODEL_DIR, "model.onnx")
    providers = ["CPUExecutionProvider"]

    self.session = InferenceSession(model_path, providers=providers)

    self.image_processor = AutoImageProcessor.from_pretrained(MODEL_DIR)

  @method()
  def inference(self, base64_image: str):

    # Decode the base64 string into raw bytes
    image_bytes = base64.b64decode(base64_image)
    pil_img = PILImage.open(BytesIO(image_bytes)).convert("RGB")

    inputs = self.image_processor(pil_img, return_tensors="np")

    ort_inputs = {"pixel_values": inputs["pixel_values"]}

    outputs = self.session.run(None, ort_inputs)
    logits = outputs[0]  


    probs = 1.0 / (1.0 + np.exp(-logits))  # (batch_size, num_labels)

    results = []
    threshold = 0.5
    for row in probs:
      row_labels = [LABELS[i] for i, score in enumerate(row) if score >= threshold]
      results.append({"scores": row.tolist(), "predicted_labels": row_labels})

    return results

  @web_endpoint(method="POST", docs=True)
  def web_inference(
    self,
    base64_image: str = Body(..., description="Base64-encoded image"),
    token: HTTPAuthorizationCredentials = Depends(auth_scheme),
  ):
  
    # bearer-token check
    if token.credentials != os.environ["AUTH_TOKEN"]:
      raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Incorrect bearer token",
        headers={"WWW-Authenticate": "Bearer"},
      )
    # Local call into .inference()
    return self.inference.local(base64_image)


@app.local_entrypoint()
def main():
  """
  Quick local test.
  Replace <BASE64_STRING> with an actual base64-encoded image for real usage.
  """
  test_b64_image = "<BASE64_STRING>"
  result = Model().inference.remote(test_b64_image)
  print("Inference result:", result)
