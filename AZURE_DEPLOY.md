# Azure Deployment Checklist (backend)

This document summarizes recommended steps to deploy the backend to Azure (App Service or Container) and required configuration for credentials and runtime.

1. Choose hosting model

- App Service (Linux/Windows): simple, runs `uvicorn` via startup command or container.
- Container (Azure Container Apps / Web App for Containers / AKS): recommended if you need custom OS or native libs (poppler) or more control.

2. Build image or runtime package

- For App Service (code): ensure `requirements-locked.txt` is used to install exact packages in deployment pipeline.
- For container: build a Docker image with Python 3.11, copy code and `requirements-locked.txt`, `pip install -r requirements-locked.txt` during image build.

3. Secrets & credentials

- Google service account JSON: store in Azure Key Vault or App Service secure file / Secret. Do NOT commit JSON to repo.
- Google client usage: set `GOOGLE_APPLICATION_CREDENTIALS` to the path where service account JSON will be mounted in the container or App Service.
- Azure Document Intelligence: set `AZURE_ENDPOINT` and `AZURE_KEY` (or `AZURE_DI_ENDPOINT`/`AZURE_DI_KEY`).
- Azure OpenAI (if used): set `AOAI_ENDPOINT`, `AOAI_API_KEY` and optional deployment name variables.

4. Recommended infra for secrets

- Use Azure Key Vault to store service account JSON and API keys; give App Service/Container a managed identity and grant access to Key Vault.
- Alternatively store minimal secrets in App Service Application Settings (encrypted at rest). Avoid storing JSON plaintext in env.

5. Startup command

- For App Service (Linux): set startup command to:
  `gunicorn -k uvicorn.workers.UvicornWorker app.main:app --bind 0.0.0.0:8000 --workers 2`
  or use `uvicorn` directly for dev: `uvicorn app.main:app --host 0.0.0.0 --port 8000`
- For container: set container command accordingly.

6. External binaries

- If you choose `pdf2image`, ensure `poppler` is installed in the host/container image (apt-get install poppler-utils).
- `PyMuPDF` is a precompiled wheel for many platforms; if not available, choose container with matching OS.

7. CI/CD (GitHub Actions -> Azure)

- Use the CI workflow created (`.github/workflows/ci.yml`) to validate and generate `requirements-locked.txt` as an artifact.
- Add a separate `deploy.yml` that builds and pushes a Docker image or uses `azure/webapps-deploy` to publish code to App Service.
- Configure GitHub Secrets: `AZURE_WEBAPP_NAME`, `AZURE_CREDENTIALS` (service principal JSON), `AZURE_RESOURCE_GROUP`, and secrets for Google/Azure keys (or reference Key Vault via managed identity).

8. Runtime configuration (App settings)

- `TEMPLATE_SPREADSHEET_ID`, `IRREGULAR_DEST_SS_ID`, `IRREGULAR_GID`, `DRIVE_FOLDER_ID` etc. from `config/settings.py` should be set as app settings or environment variables.

9. Observability & monitoring

- Enable Application Insights for logs and metrics.
- Export the `debug_logs` or ensure SSE logs are captured in App Service logs.

10. Scaling & quotas

- Be mindful of Google Drive API quotas and Azure DI/LLM rate limits. Scale App Service or use queueing for large batch jobs.

11. Post-deploy checks

- Validate endpoints: health endpoint (`/health`) returns OK.
- Upload a small sample to verify DI → LLM (if configured) → sheet export works.

Notes:

- If using containers, include `requirements-locked.txt` in the Docker image build and pin OS packages for reproducibility.
- For production, prefer `requirements-locked.txt` for deterministic installs.
