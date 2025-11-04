from google.oauth2 import service_account
from google.auth.transport.requests import Request

creds = service_account.Credentials.from_service_account_file(
    "service-account.json",
    scopes=["https://www.googleapis.com/auth/spreadsheets"],
)
creds.refresh(Request())
print("OK", creds.expiry)
