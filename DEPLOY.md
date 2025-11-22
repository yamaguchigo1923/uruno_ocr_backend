# Deployment notes

This file summarizes how to prepare a clean environment for deploying the backend without relying on the existing local virtualenv.

1. Python

- Use Python 3.10+ (the project was developed/tested on 3.11). Create a fresh virtual environment before installing requirements.

2. Install requirements
   Run in `cmd.exe`:

```
cd c:\Users\user\project\uruno_ocr_demo\backend
python -m pip install -U pip
python -m pip install -r requirements.txt
```

Notes:

- Some libraries are optional or provide alternative backends:
  - `PyMuPDF` (package name `PyMuPDF`) is preferred for PDF->image rendering. Alternative: `pdf2image` + poppler installed on the host.
  - `aiohttp` / `httpx` are used by some async clients; `openai` may use `httpx` by default.
- If your environment needs specific versions, pin them in `requirements.txt` (e.g. `openai>=1.51.0`).

3. Credentials & Environment variables

- Google Sheets / Drive: the project expects a service account JSON and the following environment vars (or `config.settings` may reference them):
  - `GOOGLE_APPLICATION_CREDENTIALS` or a path via `service-account.json` usage in settings
  - `GCS_BUCKET_NAME` (optional)
- Azure Document Intelligence:
  - `AZURE_ENDPOINT` / `AZURE_KEY` (or `AZURE_DI_ENDPOINT` / `AZURE_DI_KEY`)
  - Optionally `AZURE_DI_API_VERSION`
- Azure OpenAI (if used):
  - `AOAI_ENDPOINT`, `AOAI_API_KEY`, `AOAI_DEPLOYMENT` (depends on configuration)

4. Starting the server (development)

```
cd c:\Users\user\project\uruno_ocr_demo\backend
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

5. Optional: validating Google credentials

- The repo contains `backend/scripts/verify_google_creds.py` to validate Google API access.

6. Troubleshooting

- If imports fail after installing requirements, ensure you're using the same Python major version as used during development.
- For PDF conversion errors, install either `PyMuPDF` or `pdf2image` and (for pdf2image) the host `poppler` binaries.

7. Next steps / recommendations

- Pin versions after CI verifies a working set (e.g. `pip freeze > requirements.txt` in a controlled environment).
- Add a CI job to create a clean virtualenv and `pip install -r requirements.txt && python -m py_compile backend/**/*.py` to validate installs on each PR.
