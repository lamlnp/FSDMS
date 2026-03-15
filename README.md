# FaceService

FastAPI face recognition service for the Dormitory Management System, powered by OpenCV and InsightFace.

## What This Service Does

FaceService is a separate Python service used by the dormitory system to:

- load the face detection and embedding model
- register a student's face from an uploaded image
- start and stop camera recognition loops
- read frames from webcam or RTSP sources
- detect faces and generate 512D embeddings
- send recognition payloads back to `BEDMS`

This service does not store business data in a database. `BEDMS` remains the system of record for students, embeddings, and access logs.

## How It Fits Into The System

Runtime flow:

1. `BEDMS` calls FaceService to register faces or start cameras.
2. FaceService reads camera frames with OpenCV.
3. FaceService uses InsightFace to detect faces and generate embeddings.
4. FaceService sends detection results to the `BEDMS` callback endpoint.
5. `BEDMS` matches embeddings, creates access logs, and pushes live updates to `FEDMS`.

## Tech Stack

- Python 3.11+
- FastAPI
- Uvicorn
- OpenCV
- InsightFace
- ONNX Runtime GPU
- HTTPX

## Project Structure

```text
FaceService/
├── app/
│   ├── camera_manager.py
│   ├── config.py
│   ├── face_engine.py
│   ├── main.py
│   ├── recognition_loop.py
│   └── utils.py
├── .env.example
├── requirements.txt
└── README.md
```

## Prerequisites

Before running this service, make sure you have:

- Python 3.11 or newer
- a working webcam or RTSP camera source
- `BEDMS` running if you want real callback flow
- GPU drivers installed if you plan to use `onnxruntime-gpu`

If your machine does not support GPU inference, you may need to switch to a CPU-compatible ONNX Runtime package.

## Environment Setup

Create and activate a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Create your local env file:

```powershell
Copy-Item .env.example .env
```

Then update `.env` for your machine and local service URLs.

## Environment Variables

FaceService loads configuration from `.env`.

| Variable | Required | Default | Description |
| --- | --- | --- | --- |
| `HOST` | No | `0.0.0.0` | FastAPI bind host |
| `PORT` | No | `8000` | FastAPI port |
| `BEDMS_CALLBACK_URL` | Yes | `http://localhost:3001/v1/face-recognition/callback` | BEDMS callback endpoint |
| `FACE_SERVICE_API_KEY` | Yes | empty | Shared API key used when posting callbacks to BEDMS |
| `INSIGHTFACE_MODEL` | No | `buffalo_l` | InsightFace model name |
| `GPU_DEVICE_ID` | No | `0` | GPU device id. Use a negative value to force CPU mode in code paths that support it |
| `DET_SIZE` | No | `640` | Detection input size |
| `MIN_FACE_SIZE` | No | `50` | Minimum accepted face width/height in pixels |

## Running The Service

Start with Uvicorn:

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Or run the module directly:

```powershell
python -m app.main
```

Once started, open:

- `http://localhost:8000/health`

Expected response:

```json
{
  "status": "ok",
  "model_loaded": true,
  "service": "FaceService"
}
```

## API Endpoints

### `GET /health`

Checks whether the service is running and whether the model is loaded.

### `POST /detect`

Accepts an uploaded image and returns face bounding boxes and detection scores.

Form field:

- `image`

### `POST /register`

Accepts an uploaded image, detects the largest valid face, and returns:

- `embedding`
- `bbox`
- `det_score`
- `quality_score`
- `face_crop_base64`

This is used by `BEDMS` during student face registration.

### `POST /cameras/{camera_id}/start`

Starts a recognition loop for a configured camera source.

Example body:

```json
{
  "source_type": "webcam",
  "source_url": "0",
  "camera_type": "checkin",
  "fps_target": 5,
  "recognition_threshold": 0.6
}
```

### `POST /cameras/{camera_id}/stop`

Stops the recognition loop for the given camera.

### `GET /cameras/{camera_id}/status`

Returns current runtime status such as:

- `status`
- `fps_actual`
- `last_frame_at`
- `error_message`

### `GET /cameras`

Lists currently registered cameras known to the FaceService process.

## Integration Notes

This service is designed to work with:

- `BEDMS` for registration requests, camera control, and callback handling
- `FEDMS` indirectly through `BEDMS`

Important notes:

- FaceService keeps camera runtime state in memory.
- If FaceService restarts, active camera loops must be started again.
- The callback API key in FaceService must match `FACE_SERVICE_API_KEY` in `BEDMS`.

## Common Local Development Setup

Typical local ports:

- `FEDMS`: `5173`
- `BEDMS`: `3001`
- `FaceService`: `8000`

Typical startup order:

1. Start `BEDMS`
2. Start `FaceService`
3. Start `FEDMS`
4. Use the security or manager UI to test face registration and camera control

## Troubleshooting

### Model fails to load

Check:

- Python version
- `insightface` installation
- CUDA / GPU runtime compatibility
- `onnxruntime-gpu` support on your machine

### Callback to BEDMS fails

Check:

- `BEDMS_CALLBACK_URL`
- `FACE_SERVICE_API_KEY`
- whether `BEDMS` is running
- firewall or local port conflicts

### Camera does not open

Check:

- webcam is not already occupied by another app
- correct `source_type`
- correct `source_url`
- RTSP URL is reachable from your machine

## Git Notes

Do not commit these local-only files:

- `.env`
- `venv/`
- `__pycache__/`

Use `.env.example` as the shared template for the team.
