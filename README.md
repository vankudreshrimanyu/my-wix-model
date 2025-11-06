# Web API for Your PyTorch Model (Xception)

This bundle gives you **direct, step-by-step files** to deploy a FastAPI service for your model and connect it to your Wix site.

## 0) What to put where
- Put your trained weights file **`xception_torch_best.pt`** inside the `webapi/` folder (same level as `app.py`).
- If you don't want to commit the model to Git, upload it in your host and set `MODEL_PATH` env var accordingly.

## 1) Local run (optional)
```bash
cd webapi
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export MODEL_PATH=xception_torch_best.pt            # or set in .env
uvicorn app:app --reload
# open http://127.0.0.1:8000/health
```

## 2) Deploy (Railway / Render / etc.)
- Create a new project from this folder.
- Set environment variable: `MODEL_PATH=xception_torch_best.pt`
- Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
- After deploy, note your public URL, e.g. `https://your-app.up.railway.app`

## 3) CORS
`app.py` currently allows:
- https://xdk0aj-my-site-ia9z7xqu-vankudreshrimanyu.wix-vibe.com

If you add a custom domain later, add it to `ALLOWED_ORIGINS` in `app.py` and redeploy.

## 4) Test the endpoint
```bash
python test_request.py path/to/sample.jpg
```

## 5) Connect from Wix (two options)

### Option A: Call the API directly from Page Code (frontend)
Use the file: `wix/frontend/pageCode.js` and replace `API_BASE` with your deployed URL.

### Option B: Use Wix backend proxy
Use `wix/backend/http-functions.js` so your frontend calls `/_functions/predict` on your Wix site,
which forwards to your external API. Useful to avoid CORS issues and to hide the external URL.

## 6) Notes
- Input is a single image; the API returns `{prob_fake, label, threshold}`.
- Normalization uses ImageNet mean/std and resizes to 224×224 to match training.
