# Hosting HIMADRI

The app is a **single FastAPI service**: it serves the REST API **and** the built
React UI, and renders map layers from the precomputed rasters in
`data/outputs/real_faustini/`. So "hosting the website" = hosting that one service.

**Runtime footprint is ~17 MB** (`web/dist` 13 MB + `data/outputs/real_faustini` 4 MB).
The multi-GB raw DFSAR `.zip` / LOLA `.IMG` are **not needed at runtime** — they're
git-ignored and dockerignored. Re-running the pipeline (`himadri run`) needs them; the
deployed app only reads the precomputed outputs.

Files already prepared for you: `requirements.txt`, `Dockerfile`, `.dockerignore`, `.gitignore`.

---

## Before anything: make sure the build is fresh
```bash
cd web && npm run build && cd ..   # refreshes web/dist (only if you changed the UI)
```

---

## Option A — Instant public link (best for demo day, ~2 min)
No deploy. Run locally, expose with a Cloudflare tunnel (free, no signup):
```bash
# terminal 1 — run the app
.venv/bin/python -m uvicorn himadri.api:app --host 0.0.0.0 --port 8000

# terminal 2 — public HTTPS URL
brew install cloudflared          # one-time
cloudflared tunnel --url http://localhost:8000
```
It prints a `https://<random>.trycloudflare.com` URL you can share with judges.
(ngrok works too: `ngrok http 8000`.) Caveat: live only while your laptop runs it.

---

## Option B — Render.com (free, persistent, recommended)
1. Push the repo to GitHub (big files are git-ignored, so this stays small):
   ```bash
   git add -A && git commit -m "HIMADRI web app + deploy files" && git push
   ```
2. On render.com → **New → Web Service** → connect the repo.
3. Render auto-detects the **Dockerfile**. Settings:
   - Environment: **Docker**
   - Health check path: `/api/health`
   - Instance: Free tier is fine.
4. Deploy → you get a public `https://<name>.onrender.com` URL.
   (`$PORT` is injected automatically; the Dockerfile already uses it.)

---

## Option C — Any VPS (DigitalOcean / EC2 / Lightsail) with Docker
```bash
# on the server, after git clone (or scp the repo):
docker build -t himadri .
docker run -d --restart unless-stopped -p 80:8000 himadri
# open http://<server-ip>/
```
No-Docker variant: install Python 3.12, then
```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
PYTHONPATH=src .venv/bin/uvicorn himadri.api:app --host 0.0.0.0 --port 8000
```
Put nginx in front for a domain + HTTPS (Let's Encrypt) if you want a custom URL.

---

## Option D — Hugging Face Spaces (free, Docker SDK)
1. Create a **Space** → SDK: **Docker**.
2. Push the repo to the Space. It builds the Dockerfile.
3. HF serves on port **7860** — either set a Space variable `PORT=7860`, or add
   `EXPOSE 7860` and it reads `$PORT`. Public URL is provided automatically.

---

## Local Docker (to test before deploying)
Start Docker Desktop, then:
```bash
docker build -t himadri .
docker run --rm -p 8000:8000 himadri
# http://localhost:8000   ·   health: http://localhost:8000/api/health
```

## Notes
- The synthetic **Model & Validation** tab computes on first request (~10–15 s) then caches.
- To update the real scene shown, re-run `himadri run --config config/real_faustini.yaml`
  locally, commit the refreshed `data/outputs/real_faustini/`, and redeploy.
- CORS isn't needed — the API and UI are served from the same origin.
