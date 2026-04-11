# KYC Risk Scoring — Flask API + Svelte Dashboard

## Stack

- Backend: Flask API in `backend/app.py`
- Pipeline: `backend/pipeline.py`
- Async run manager: `backend/job_manager.py`
- Frontend: Svelte + Vite in `frontend/`

## Run backend

1. `cd backend`
2. `python -m venv .venv && source .venv/bin/activate`
3. `pip install -r requirements.txt`
4. `python app.py`

Backend runs on `http://localhost:5000`.

## Run frontend

1. `cd frontend`
2. `npm install`
3. `npm run dev`

Frontend runs on `http://localhost:5173` and proxies `/api` to backend.

## API quick check

- `GET /api/health`
- `POST /api/runs`
- `GET /api/runs/{run_id}`
- `GET /api/runs/{run_id}/summary`
- `GET /api/runs/{run_id}/artifacts`
- `POST /api/score`
