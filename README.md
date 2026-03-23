# experience_day_deployment

YOLO inference service packaged for Amazon SageMaker endpoint deployment.

## Inference endpoints

- `GET /ping` - health check endpoint used by SageMaker.
- `POST /invocations` - primary SageMaker inference endpoint.
- `POST /detect` - backward-compatible alias for existing clients.
