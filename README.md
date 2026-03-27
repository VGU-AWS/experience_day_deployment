# experience_day_deployment

SegFormer (NVIDIA, Hugging Face) inference service packaged for Amazon SageMaker endpoint deployment.

## Inference endpoints

- `GET /ping` - health check endpoint used by SageMaker.
- `POST /invocations` - primary SageMaker inference endpoint.
- `POST /detect` - backward-compatible alias for existing clients.

## Container runtime

- Base image: `nvidia/cuda:12.1.1-cudnn9-runtime-ubuntu22.04` with PyTorch (`torch` cu121 wheel) installed during build for GPU-backed web/SageMaker endpoint serving.

## Model

- Default model: `nvidia/segformer-b1-finetuned-ade-512-512`
- Override with environment variable: `SEGFORMER_MODEL`
- Human class extraction uses ADE20K class id `12` (`person`)
