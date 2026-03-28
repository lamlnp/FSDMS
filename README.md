# FaceService

FaceService is a separate Python service for face detection, face embedding, and camera-based recognition in the Dormitory Management System.

It uses:

- FastAPI for HTTP endpoints
- OpenCV for image and camera handling
- InsightFace for face detection and 512D embeddings
- ONNX Runtime for inference

This service is intentionally stateless for business data. It does not store student records or access logs in a database. `BEDMS` remains the system of record.

## What This Service Does

FaceService is responsible for:

- loading the face detection and embedding model at startup
- detecting faces in uploaded images
- extracting a 512D embedding from the largest valid face in an image
- starting and stopping camera capture loops
- reading frames from webcams or RTSP streams
- sending recognition results back to `BEDMS`

## What This Service Does Not Do

FaceService does not:

- store students, cards, or access logs permanently
- decide access control policy by itself
- replace `BEDMS`
- keep camera state after the process restarts

If FaceService restarts, camera loops must be started again.

## Runtime Modes

The code supports two practical machine profiles:

- NVIDIA GPU machine
- CPU-only machine

The runtime behavior is controlled mainly by `GPU_DEVICE_ID` in `.env`.

| `GPU_DEVICE_ID` | Runtime mode | Provider behavior |
| --- | --- | --- |
| `0` or any non-negative value | GPU mode | Tries `CUDAExecutionProvider` first, then `CPUExecutionProvider` |
| `-1` or any negative value | CPU mode | Uses `CPUExecutionProvider` only |

Important:

- The repository currently lists `onnxruntime-gpu` in `requirements.txt`.
- On CPU-only machines, that package may still install, but if it causes CUDA or provider issues, switch to the CPU runtime package locally by uninstalling `onnxruntime-gpu` and installing `onnxruntime`.
- Do not commit local dependency swaps unless the team wants to standardize the project for CPU-only deployment.

## How The Service Fits Into The System

Typical flow:

1. `BEDMS` asks FaceService to register a face or start a camera.
2. FaceService reads frames or processes an uploaded image.
3. FaceService detects the face and generates an embedding.
4. FaceService returns the result immediately for synchronous endpoints.
5. For real-time camera recognition, FaceService posts recognition data back to the `BEDMS` callback endpoint.
6. `BEDMS` matches the embedding, writes business data, and can forward live updates to `FEDMS`.

## Repository Layout

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

## Requirements

### Minimum

- Python 3.11 or newer
- `pip`
- access to the `FaceService` source code
- a webcam or RTSP source if you want camera features

### Optional but recommended

- NVIDIA GPU for faster inference
- working `BEDMS` instance for callback testing
- a sample image file for verifying `/detect` and `/register`

### If You Have An NVIDIA GPU

Use this path if your machine has:

- an NVIDIA GPU
- the required NVIDIA driver stack
- a Python environment that can load CUDA-enabled ONNX Runtime

GPU mode is the default expected by the current `requirements.txt`.

### If You Do Not Have An NVIDIA GPU

Use CPU mode by setting:

- `GPU_DEVICE_ID=-1`

If the GPU runtime package causes problems on your machine, install the CPU runtime package locally:

```powershell
pip uninstall -y onnxruntime-gpu
pip install onnxruntime
```

Then keep `GPU_DEVICE_ID=-1` in `.env`.

## Quick Start

The following steps are the most direct way to get FaceService running on a new machine.

### 1. Open a terminal in `FaceService`

```powershell
cd FaceService
```

### 2. Create a virtual environment

```powershell
python -m venv venv
```

### 3. Activate the virtual environment

On Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

On Windows Command Prompt:

```bat
venv\Scripts\activate.bat
```

On macOS or Linux:

```bash
source venv/bin/activate
```

### 4. Install dependencies

#### GPU machine

```powershell
pip install -r requirements.txt
```

#### CPU-only machine

Install the default dependencies first:

```powershell
pip install -r requirements.txt
```

If GPU runtime loading fails, switch to the CPU runtime:

```powershell
pip uninstall -y onnxruntime-gpu
pip install onnxruntime
```

### 5. Create your local `.env`

```powershell
Copy-Item .env.example .env
```

Edit `.env` for your machine.

### 6. Start the service

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Or run the module directly:

```powershell
python -m app.main
```

### 7. Verify it started

Open:

- `http://localhost:8000/health`
- `http://localhost:8000/docs`

If startup is successful, `/health` should return `model_loaded: true`.

## Environment Setup

FaceService reads configuration from a local `.env` file.

Use `.env.example` as the base template.

```powershell
Copy-Item .env.example .env
```

Then edit the values to match your environment.

## Environment Variables

| Variable | Required | Default | Description |
| --- | --- | --- | --- |
| `HOST` | No | `0.0.0.0` | Host address used by the FastAPI server |
| `PORT` | No | `8000` | Port used by the FastAPI server |
| `BEDMS_CALLBACK_URL` | Yes for camera callback flow | `http://localhost:3001/v1/face-recognition/callback` | Endpoint that FaceService posts recognition results to |
| `FACE_SERVICE_API_KEY` | Yes for callback authentication | empty | Shared API key sent in the `X-API-Key` header when posting to BEDMS |
| `INSIGHTFACE_MODEL` | No | `buffalo_l` | InsightFace model package name |
| `GPU_DEVICE_ID` | No | `0` | GPU device index. Use `-1` for CPU-only mode |
| `DET_SIZE` | No | `640` | Face detection input size. Larger values may improve detection but use more CPU or GPU |
| `MIN_FACE_SIZE` | No | `50` | Minimum accepted face width and height in pixels |

## Configuration Notes

### `GPU_DEVICE_ID`

This is the most important setting for machine compatibility.

- Set `GPU_DEVICE_ID=0` on a machine with a working NVIDIA GPU if you want GPU inference.
- Set `GPU_DEVICE_ID=-1` on a CPU-only machine.

The code in `FaceService/app/face_engine.py` selects providers based on this value.

### `BEDMS_CALLBACK_URL`

This should point to the BEDMS callback endpoint that receives recognition events.

If BEDMS is running locally, the default value is usually correct:

```text
http://localhost:3001/v1/face-recognition/callback
```

If BEDMS is running on another machine, update the host name or IP address.

### `FACE_SERVICE_API_KEY`

This value must match the API key that BEDMS expects for FaceService callbacks.

If the two services use different keys, callbacks will fail even if the face recognition itself works.

### `INSIGHTFACE_MODEL`

The code defaults to `buffalo_l`.

If the team changes the model package later, update this value and keep the README in sync.

### `DET_SIZE`

This controls the detection input resolution.

- Higher values can improve detection on small faces
- Higher values also increase compute cost

If recognition feels too slow on a CPU-only machine, this is one of the first values to reduce.

### `MIN_FACE_SIZE`

Faces smaller than this threshold are ignored.

This helps avoid storing or processing tiny, low-quality detections.

## First Run Behavior

The first startup can be slower than later startups because the InsightFace model may be downloaded and cached.

Expect the following on first run:

- model download if the cache is empty
- longer startup time than normal
- additional console logs while the model is loaded

If the machine is offline and the model is not already cached, startup may fail until the model can be downloaded once.

## Windows-Specific Notes

The code includes a Windows helper in `FaceService/app/face_engine.py` that tries to register CUDA DLL directories from the local virtual environment before importing ONNX Runtime and InsightFace.

That means:

- the standard `venv` layout is expected
- GPU wheels installed through `pip` should be discoverable when they place DLLs under `venv\Lib\site-packages\nvidia`
- if you install CUDA libraries in a non-standard location, make sure they are available through `PATH`

If you are on a CPU-only Windows machine, you can ignore the CUDA DLL logic and run with `GPU_DEVICE_ID=-1`.

## Running The Service

Start the service from the `FaceService` directory.

### Option 1: Uvicorn command

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Option 2: Python module

```powershell
python -m app.main
```

The `python -m app.main` path uses the `HOST` and `PORT` values from `.env`.

## Health Check

Call:

```text
GET /health
```

Example response:

```json
{
  "status": "ok",
  "model_loaded": true,
  "service": "FaceService"
}
```

What it means:

- `status: ok` means the API process is running
- `model_loaded: true` means the InsightFace model initialized successfully
- `service` identifies the service name

If `model_loaded` is `false`, check the startup logs first.

## API Endpoints

### `GET /health`

Checks whether the service is up and whether the face model has loaded.

Use this as the first verification step after startup.

### `POST /detect`

Detects faces in an uploaded image and returns bounding boxes plus detection scores.

Request type:

- `multipart/form-data`

Form field:

- `image`

Response data:

- `success`
- `faces_count`
- `detections`

This endpoint does not return embeddings.

### `POST /register`

Detects the largest valid face in the uploaded image and returns a 512D embedding.

Request type:

- `multipart/form-data`

Form field:

- `image`

Response data:

- `success`
- `embedding`
- `bbox`
- `det_score`
- `quality_score`
- `face_crop_base64`

Behavior notes:

- If no valid face is found, the service returns HTTP `422`
- `quality_score` is the same value as `det_score`
- `face_crop_base64` is a JPEG-encoded crop of the detected face

### `POST /cameras/{camera_id}/start`

Starts a camera capture loop and a recognition loop for the given camera ID.

The request body is JSON.

Example:

```json
{
  "source_type": "webcam",
  "source_url": "0",
  "camera_type": "checkin",
  "fps_target": 5,
  "recognition_threshold": 0.6
}
```

Field details:

- `source_type` must be `webcam` or `rtsp`
- `source_url` is the webcam index as a string, or an RTSP URL
- `camera_type` must be `checkin` or `checkout`
- `fps_target` must be between `1` and `30`
- `recognition_threshold` must be between `0.0` and `1.0`

### `POST /cameras/{camera_id}/stop`

Stops the capture loop and recognition loop for the specified camera.

### `GET /cameras/{camera_id}/status`

Returns runtime information for a specific camera.

Useful fields:

- `status`
- `camera_type`
- `source_type`
- `fps_target`
- `fps_actual`
- `last_frame_at`
- `error_message`

### `GET /cameras`

Lists all cameras currently registered in the running FaceService process.

## Example Requests

### Check health

```powershell
curl.exe http://localhost:8000/health
```

### Test face detection

```powershell
curl.exe -F "image=@sample.jpg" http://localhost:8000/detect
```

### Test face registration

```powershell
curl.exe -F "image=@sample.jpg" http://localhost:8000/register
```

### Start a webcam camera

```powershell
$body = @{
  source_type = "webcam"
  source_url = "0"
  camera_type = "checkin"
  fps_target = 5
  recognition_threshold = 0.6
} | ConvertTo-Json

Invoke-RestMethod -Method Post `
  -Uri "http://localhost:8000/cameras/cam-1/start" `
  -ContentType "application/json" `
  -Body $body
```

### Check camera status

```powershell
curl.exe http://localhost:8000/cameras/cam-1/status
```

## Integration With BEDMS

FaceService is meant to be used together with `BEDMS`.

Integration rules:

- FaceService sends recognition callbacks to `BEDMS_CALLBACK_URL`
- callbacks include the `X-API-Key` header using `FACE_SERVICE_API_KEY`
- `BEDMS` should be running before you try to exercise the callback path
- FaceService does not persist the callback history itself

If the callback URL or API key is wrong, the face service can still load and answer `/detect` and `/register`, but camera recognition integration will fail later when it tries to post results.

## Local Development Order

Typical startup order for the full system:

1. Start `BEDMS`
2. Start `FaceService`
3. Start `FEDMS`
4. Use the manager or security UI to test registration and camera control

Typical local ports:

- `FEDMS`: `5173`
- `BEDMS`: `3001`
- `FaceService`: `8000`

## Validation Checklist

Use this list when helping someone set up the service.

1. Python 3.11+ is installed.
2. The virtual environment is created and activated.
3. Dependencies installed successfully.
4. `.env` exists and values were reviewed.
5. `GPU_DEVICE_ID` matches the machine type.
6. `BEDMS_CALLBACK_URL` points to a running BEDMS instance if callback flow is needed.
7. `FACE_SERVICE_API_KEY` matches BEDMS.
8. `GET /health` returns `status: ok`.
9. `model_loaded` is `true`.
10. `GET /docs` opens successfully.
11. A sample image works with `POST /detect` or `POST /register`.
12. Camera start and stop work if camera features are in scope.

## Troubleshooting

### `/health` is reachable but `model_loaded` is `false`

Likely causes:

- model load failed during startup
- GPU provider is not available
- dependency mismatch in `onnxruntime`, `onnxruntime-gpu`, or `insightface`
- the machine is missing the required runtime libraries

What to check:

- the console logs printed during startup
- the value of `GPU_DEVICE_ID`
- whether the machine is supposed to use GPU or CPU mode

### Startup fails on a CPU-only machine

Try this sequence:

1. Set `GPU_DEVICE_ID=-1`
2. Uninstall `onnxruntime-gpu`
3. Install `onnxruntime`
4. Restart the service

If the model loads after that, the problem was GPU runtime selection rather than the model itself.

### `POST /detect` returns `Invalid image file`

The uploaded payload is not a valid image or the `image` field name is wrong.

Make sure the request is sent as `multipart/form-data` with a file field named `image`.

### `POST /register` returns HTTP `422`

The image was valid, but no usable face was detected.

Common causes:

- the face is too small
- the image is blurry
- the person is turned away
- the lighting is poor
- the image contains multiple faces and none is large enough

### Camera does not open

Check:

- `source_type` is correct
- `source_url` is correct
- the webcam is not already in use by another application
- the RTSP URL is reachable from this machine
- the camera permissions are allowed on the operating system

### Recognition callbacks fail

Check:

- `BEDMS` is running
- `BEDMS_CALLBACK_URL` is correct
- `FACE_SERVICE_API_KEY` matches the value expected by `BEDMS`
- local firewall rules are not blocking the connection
- the endpoint path in BEDMS has not changed

### Camera status disappears after restart

This is expected.

Camera runtime state is stored in memory only. If the FaceService process restarts, you must call the start endpoint again.

## Notes For Team Members Using AI To Help Set Up The Service

If you ask an AI assistant to help configure this service, give it these facts up front:

- this repo contains a separate FaceService process
- the service is started from the `FaceService` directory
- GPU mode is optional
- CPU-only machines should use `GPU_DEVICE_ID=-1`
- the current requirements file uses `onnxruntime-gpu`
- the service must load its model successfully before it can serve face requests
- `BEDMS` is required only for callback and integration testing, not for basic `/health`, `/detect`, or `/register` checks

That context should be enough for an assistant to help someone diagnose a failed setup without guessing at the architecture.

## Git Notes

Do not commit local-only files:

- `.env`
- `venv/`
- `__pycache__/`

Use `.env.example` as the shared template for the team.
