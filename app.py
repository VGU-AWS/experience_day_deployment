from fastapi import FastAPI, HTTPException, Request, Response
import base64
from io import BytesIO
import os
from threading import Lock
import uvicorn
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation


# Detect GPU availability
if torch.cuda.is_available():
    if torch.cuda.get_device_name().startswith("NVIDIA"):
        device = torch.device("cuda")
        print("Using NVIDIA GPU with CUDA")
    elif torch.cuda.get_device_name().startswith("gfx"):
        device = torch.device("cuda")
        print("Using AMD GPU with ROCm")
    else:
        device = torch.device("cpu")
        print("CUDA device detected but not supported, using CPU")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple GPU with Metal Performance Shaders (MPS)")
else:
    device = torch.device("cpu")
    print("No GPU available, using CPU")

app = FastAPI()
segformer_model_name = os.getenv("SEGFORMER_MODEL", "nvidia/segformer-b1-finetuned-ade-512-512")
segformer_processor = None
segformer_model = None
segformer_lock = Lock()


def _get_segformer() -> tuple[AutoImageProcessor, SegformerForSemanticSegmentation]:
    global segformer_processor, segformer_model

    if segformer_processor is None or segformer_model is None:
        with segformer_lock:
            if segformer_processor is None or segformer_model is None:
                segformer_processor = AutoImageProcessor.from_pretrained(segformer_model_name)
                segformer_model = SegformerForSemanticSegmentation.from_pretrained(segformer_model_name)
                segformer_model.to(device)
                segformer_model.eval()

    return segformer_processor, segformer_model


def _encode_png_mask(mask: np.ndarray) -> str:
    try:
        image = Image.fromarray(mask)
        buffer = BytesIO()
        image.save(buffer, format="PNG")
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to encode mask image") from exc

    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _empty_response(image_shape: tuple[int, int]) -> dict:
    empty_mask = np.zeros(image_shape, dtype=np.uint8)
    inverse_mask = np.ones(image_shape, dtype=np.uint8) * 255
    return {
        "mask": _encode_png_mask(empty_mask),
        "mask_inv": _encode_png_mask(inverse_mask),
        "locations": [],
    }


def _location_from_mask(person_mask: np.ndarray) -> list[list[int]]:
    ys, xs = np.where(person_mask)
    if xs.size == 0 or ys.size == 0:
        return []

    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max())
    y2 = int(ys.max())
    return [[x1, y1, x2, y2]]


# health check
@app.get("/ping")
def ping():
    return Response(status_code=200)


@app.post("/invocations")
async def invocations(request: Request):
    image_bytes = await request.body()
    try:
        pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image payload")

    img = np.array(pil_image)
    processor, model = _get_segformer()

    with torch.no_grad():
        inputs = processor(images=pil_image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        outputs = model(pixel_values=pixel_values)
        segmentation = processor.post_process_semantic_segmentation(
            outputs,
            target_sizes=[img.shape[:2]],
        )[0]

    # ADE20K person class id used by NVIDIA SegFormer checkpoints.
    person_class_id = 12
    person_mask = (segmentation == person_class_id).to(torch.uint8) * 255

    if torch.count_nonzero(person_mask).item() == 0:
        return _empty_response(img.shape[:2])

    combined_mask_np = person_mask.cpu().numpy().astype(np.uint8)
    inv_mask_np = np.where(combined_mask_np == 0, 255, 0).astype(np.uint8)
    locations = _location_from_mask(combined_mask_np > 0)

    return {
        "mask": _encode_png_mask(combined_mask_np),
        "mask_inv": _encode_png_mask(inv_mask_np),
        "locations": locations,
    }


@app.post("/detect")
async def detect(request: Request):
    """Backward-compatible alias for existing clients."""
    return await invocations(request)


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8080")),
    )
