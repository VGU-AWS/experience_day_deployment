# experience_day_deployment

SegFormer (NVIDIA, Hugging Face) inference service packaged for Amazon SageMaker endpoint deployment.

## Inference endpoints

- `GET /ping` - health check endpoint used by SageMaker.
- `POST /invocations` - primary SageMaker inference endpoint.
- `POST /detect` - backward-compatible alias for existing clients.

## Container runtime

- Base image: `pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime` (known stable CUDA runtime on Ubuntu for GPU-backed web/SageMaker endpoint serving).

## Model

- Default model: `nvidia/segformer-b1-finetuned-ade-512-512`
- Override with environment variable: `SEGFORMER_MODEL`
- Human class extraction uses ADE20K class id `12` (`person`)
