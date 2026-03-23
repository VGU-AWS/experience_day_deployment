from fastapi import FastAPI, HTTPException, UploadFile, Request, Response
import base64
from io import BytesIO
import uvicorn
import numpy as np
import supervision as sv
import torch
import torch.nn.functional as F
from PIL import Image
from ultralytics import YOLO

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
byte_tracker = sv.ByteTrack()
byte_tracker.reset()
seg_model_name = "yolo11l-seg"
seg_model = YOLO(f"{seg_model_name}.pt")


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

#health check
@app.get("/ping")
def ping():
    return Response(status_code=200)

@app.post("/invocations")
async def invocations(request: Request):
    image_bytes = await request.body()
    try:
        img = np.array(Image.open(BytesIO(image_bytes)).convert("RGB"))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image payload")

    results = seg_model(img)
    boxes = results[0].boxes
    masks = results[0].masks

    if boxes is None or len(boxes) == 0:
        return _empty_response(img.shape[:2])

    class_ids = boxes.cls
    scores = boxes.conf
    xyxys = boxes.xyxy

    confidence_threshold = 0.7
    human_class_id = 0
    valid_mask = (scores > confidence_threshold) & (class_ids == human_class_id)
    valid_indices = torch.where(valid_mask)[0]

    if valid_indices.numel() == 0 or masks is None:
        return _empty_response(img.shape[:2])

    filtered_class_ids = class_ids[valid_indices].to(torch.int16)
    filtered_scores = scores[valid_indices]
    filtered_xyxys = xyxys[valid_indices]
    filtered_masks = masks.data[valid_indices]

    detections = sv.Detections(
        xyxy=filtered_xyxys.cpu().numpy(),
        confidence=filtered_scores.cpu().numpy(),
        class_id=filtered_class_ids.cpu().numpy(),
    )

    tracked_detections = byte_tracker.update_with_detections(detections)
    masks_resized = F.interpolate(
        filtered_masks.unsqueeze(1).float(),
        size=img.shape[:2],
        mode="nearest",
    ).squeeze(1)

    combined_mask = torch.max(masks_resized, dim=0).values.to(torch.uint8) * 255
    combined_mask_np = combined_mask.cpu().numpy()
    inv_mask_np = ((combined_mask == 0).to(torch.uint8) * 255).cpu().numpy()

    return {
        "mask": _encode_png_mask(combined_mask_np),
        "mask_inv": _encode_png_mask(inv_mask_np),
        "locations": tracked_detections.xyxy.tolist(),
    }


@app.post("/detect")
async def detect(file: UploadFile):
    """Backward-compatible alias for existing clients."""
    return await invocations(file)

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8080")),
    )
