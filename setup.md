# FSDMS — Windows Setup From Scratch

Step-by-step guide to get FSDMS running on a clean Windows machine. Targets Windows 10 / 11 with PowerShell.

For deeper reference (API endpoints, env vars, troubleshooting matrix), see [README.md](README.md).

---

## 1. Prerequisites

Install these before anything else.

### 1.1 Python 3.11 or newer

1. Download from <https://www.python.org/downloads/windows/>
2. During install, check **"Add python.exe to PATH"**
3. Verify in a new PowerShell window:

   ```powershell
   python --version
   pip --version
   ```

### 1.2 Git (optional, only if cloning)

Download from <https://git-scm.com/download/win>.

### 1.3 Visual C++ Redistributable

Required by `onnxruntime` and `opencv-python`. Most Windows machines already have it. If install fails later with DLL errors, install:

- <https://aka.ms/vs/17/release/vc_redist.x64.exe>

### 1.4 NVIDIA GPU stack (GPU mode only — skip if CPU-only)

Only needed if you want CUDA inference.

1. **NVIDIA driver** — install/update from <https://www.nvidia.com/Download/index.aspx>
2. Verify the driver is alive:

   ```powershell
   nvidia-smi
   ```

3. CUDA / cuDNN system install is **not required** — the `onnxruntime-gpu` wheel bundles the DLLs it needs, and FSDMS registers them automatically on Windows from `venv\Lib\site-packages\nvidia`.

If you do not have an NVIDIA GPU, skip this section and use CPU mode (covered below).

### 1.5 ngrok (optional — only for the demo / remote callback flow)

Only needed if BEDMS (Railway) must reach your local FSDMS.

1. Install: <https://ngrok.com/download>
2. Authenticate once with your token:

   ```powershell
   ngrok config add-authtoken <YOUR_TOKEN>
   ```

---

## 2. Get The Source

If the repo is already on disk, skip to step 3.

```powershell
cd D:\
git clone <repo-url> FSDMS
cd FSDMS
```

If you already have the repo:

```powershell
cd <path-to>\FSDMS
```

---

## Fast Path: One-Shot Setup Script

If you just want everything ready, run:

```powershell
.\setup.ps1
```

The script verifies Python, probes for an NVIDIA GPU, asks whether to use GPU or CPU mode, creates the venv, installs dependencies (swapping `onnxruntime-gpu` → `onnxruntime` automatically in CPU mode), and creates `.env` from the template with `GPU_DEVICE_ID` set correctly.

Useful flags:

```powershell
.\setup.ps1 -Mode gpu        # skip the prompt, force GPU deps
.\setup.ps1 -Mode cpu        # skip the prompt, force CPU deps
.\setup.ps1 -SkipDeps        # reuse venv, only re-run .env patching
.\setup.ps1 -Force            # delete and recreate venv
```

After it finishes, edit `.env` to set `FACE_SERVICE_API_KEY` (must match BEDMS), then jump to [step 6](#6-first-run).

The manual steps below (3–5) are kept as a reference and as the fallback if the script fails.

---

## 3. Create And Activate The Virtual Environment

From inside `FSDMS\`:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If activation is blocked by execution policy, run **once per user**:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

Then re-run the activate command. Your prompt should now show `(venv)`.

Upgrade pip inside the venv:

```powershell
python -m pip install --upgrade pip
```

---

## 4. Install Dependencies

### 4.1 GPU machine (NVIDIA)

```powershell
pip install -r requirements.txt
```

This installs `onnxruntime-gpu` along with InsightFace and the rest.

### 4.2 CPU-only machine

Install everything first:

```powershell
pip install -r requirements.txt
```

Then swap the GPU runtime for the CPU runtime:

```powershell
pip uninstall -y onnxruntime-gpu
pip install onnxruntime
```

You will also set `GPU_DEVICE_ID=-1` in `.env` (next step).

> Do not commit this swap. `requirements.txt` stays on `onnxruntime-gpu` for the team.

---

## 5. Create `.env`

```powershell
Copy-Item .env.example .env
notepad .env
```

Fill in the values that matter for your setup:

| Variable | What to set |
| --- | --- |
| `HOST` | Leave `0.0.0.0` |
| `PORT` | Leave `8000` unless taken |
| `CORS_ALLOW_ORIGINS` | `https://fedms.vercel.app` for deployed FEDMS, or `http://localhost:5173` for local FEDMS (comma-separate to allow both) |
| `BEDMS_CALLBACK_URL` | Deployed: `https://bedms-production.up.railway.app/v1/face-recognition/callback`. Local BEDMS: `http://localhost:3001/v1/face-recognition/callback` |
| `FACE_SERVICE_API_KEY` | **Must match** the value in BEDMS `.env` |
| `GPU_DEVICE_ID` | `0` for GPU, `-1` for CPU |
| `INSIGHTFACE_MODEL` | Leave `buffalo_l` |
| `DET_SIZE` | Leave `640`. Lower (e.g. `320`) on slow CPU |
| `MIN_FACE_SIZE` | Leave `50` |

Save and close.

---

## 6. First Run

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

First startup downloads the InsightFace `buffalo_l` model (~280 MB) into `%USERPROFILE%\.insightface\`. Expect 30–120 s the first time. Subsequent starts are fast.

You should see logs ending with something similar to:

```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

---

## 7. Verify

In a second PowerShell window:

```powershell
curl.exe http://localhost:8000/health
```

Expected:

```json
{"status":"ok","model_loaded":true,"service":"FSDMS"}
```

Open the interactive docs in a browser:

- <http://localhost:8000/docs>

If `model_loaded` is `false`, jump to [Troubleshooting](#troubleshooting).

---

## 8. (Optional) Run With ngrok For Remote Callbacks

Used when BEDMS is on Railway and needs to call your local FSDMS (or your local browser must hit FSDMS through HTTPS).

```powershell
.\start.ps1
```

This script:

1. Verifies `.env` exists
2. Boots Uvicorn on the configured `HOST`/`PORT`
3. Spawns `ngrok http 8000` with the JWT-validation policy in [ngrok.jwt-validation.yml](ngrok.jwt-validation.yml)

To run **without** ngrok:

```powershell
.\start.ps1 -NoNgrok
```

Copy the ngrok HTTPS URL printed in the ngrok window into BEDMS as the FSDMS base URL.

---

## 9. Smoke Test With A Real Image

Place a photo with a clear face as `sample.jpg` in `FSDMS\`.

```powershell
curl.exe -F "image=@sample.jpg" http://localhost:8000/detect
curl.exe -F "image=@sample.jpg" http://localhost:8000/register
```

`/detect` returns bounding boxes. `/register` returns a 512D embedding plus a base64 face crop.

---

## Troubleshooting

### `Activate.ps1 cannot be loaded because running scripts is disabled`

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

### `python` not found after install

Close and reopen PowerShell. If still missing, re-run the Python installer and tick **Add to PATH**.

### `pip install` fails on `insightface`

Usually a missing C++ build toolchain for an older Python. Use Python 3.11+ (prebuilt wheels exist) and install VC++ Redistributable (step 1.3).

### `/health` returns `model_loaded: false`

1. Check console logs for the actual exception during startup.
2. On CPU-only machines, run the swap from step 4.2 and set `GPU_DEVICE_ID=-1`.
3. On GPU machines, confirm `nvidia-smi` works and `onnxruntime-gpu` is the installed package (`pip list | findstr onnxruntime`).

### `onnxruntime` cannot find CUDA DLLs (GPU mode)

`app\face_engine.py` adds `venv\Lib\site-packages\nvidia\**` to the DLL search path on Windows. If you installed packages outside the venv or moved them, either reinstall inside the venv or add the DLL folder to `PATH`.

### Port 8000 already in use

Change `PORT` in `.env`, or:

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8010 --reload
```

### Recognition callbacks to BEDMS fail

- Confirm `BEDMS_CALLBACK_URL` is reachable from this machine
- Confirm `FACE_SERVICE_API_KEY` matches the BEDMS value exactly
- Check Windows Defender Firewall is not blocking outbound on the chosen port

For more failure modes, see the **Troubleshooting** section in [README.md](README.md).

---

## Daily Use After Initial Setup

```powershell
cd <path-to>\FSDMS
.\venv\Scripts\Activate.ps1
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Or just:

```powershell
.\start.ps1 -NoNgrok
```
