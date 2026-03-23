from fastapi import FastAPI, UploadFile, Response
import cv2
import numpy as np
from ultralytics import YOLO
import torch
import supervision as sv
import torch.nn.functional as F
import base64

# Detect GPU availability
if torch.cuda.is_available():
    if torch.cuda.get_device_name().startswith('NVIDIA'):
        device = torch.device('cuda')
        print("Using NVIDIA GPU with CUDA")
    elif torch.cuda.get_device_name().startswith('gfx'):
        device = torch.device('cuda')
        print("Using AMD GPU with ROCm")
    else:
        device = torch.device('cpu')
        print("CUDA device detected but not supported, using CPU")
elif torch.backends.mps.is_available():
    device = torch.device('mps')
    print("Using Apple GPU with Metal Performance Shaders (MPS)")
else:
    device = torch.device('cpu')
    print("No GPU available, using CPU")

app = FastAPI()
byte_tracker = sv.ByteTrack()
byte_tracker.reset()
seg_model_name = 'yolo11l-seg'
seg_model = YOLO(f'{seg_model_name}.pt')

#health status
@app.post("/ping")
def ping():
    return Response(status_code=200)

@app.post("/detect")
async def detect(file: UploadFile):
    image_bytes = await file.read()

    img = cv2.imdecode(
        np.frombuffer(image_bytes, np.uint8),
        cv2.IMREAD_COLOR
    )

    results = seg_model(img)
    boxes = results[0].boxes  # Accessing the boxes from results
    masks = results[0].masks  # Accessing the masks from results

    if boxes is None or len(boxes) == 0:
        return {"mask": np.zeros(img.shape[:2], dtype=np.uint8).tolist(),"mask_inv":np.ones(img.shape[:2], dtype=np.uint8).tolist(), "locations": []}

    class_ids = boxes.cls
    scores = boxes.conf
    xyxys = boxes.xyxy

    confidence_threshold = 0.7
    human_class_id = 0
    valid_mask = (scores > confidence_threshold) & (class_ids == human_class_id)
    valid_indices = torch.where(valid_mask)[0]

    if valid_indices.numel() == 0:
        return {"mask": np.zeros(img.shape[:2], dtype=np.uint8).tolist(),"mask_inv":np.ones(img.shape[:2], dtype=np.uint8).tolist(), "locations": []}

    # Apply filtering to boxes and masks
    filtered_class_ids = class_ids[valid_indices].to(torch.int16)
    filtered_scores = scores[valid_indices]
    filtered_xyxys = xyxys[valid_indices]
    filtered_masks = masks.data[valid_indices] if masks is not None else []

    # Create filtered detections for tracking
    detections = sv.Detections(
        xyxy=filtered_xyxys.cpu().numpy(),
        confidence=filtered_scores.cpu().numpy(),
        class_id=filtered_class_ids.cpu().numpy(),
    )

    tracked_detections = byte_tracker.update_with_detections(detections)
    # Vectorize resize: [N, 1, H_orig, W_orig] -> [N, 1, 448, 640]
    masks_resized = F.interpolate(filtered_masks.unsqueeze(1).float(),
                    size=img.shape[::2] , mode="nearest").squeeze(1)

    # COMBINE N masks into ONE mask using max (logical OR for binary masks) and convert base64 for easy transport
    combined_mask = (torch.max(masks_resized, dim=0).value).astype(torch.uint8)*255    
    _ , mask_buf = cv2.imencode(".png",combined_mask.cpu().numpy())
    mask_base64 = base64.b64encode(mask_buf).decode("utf-8")

    inv_mask = ((combined_mask==0).astype(torch.uint8)*255).cpu().numpy()
    _ , mask_buf = cv2.imencode(".png",inv_mask)
    inv_mask_base64 = base64.b64encode(mask_buf).decode("utf-8")    

    return {"mask": mask_base64, "mask_inv": inv_mask_base64, "locations": tracked_detections.xyxy.cpu().numpy().tolist()}
