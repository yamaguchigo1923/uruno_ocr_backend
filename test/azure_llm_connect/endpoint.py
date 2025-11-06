"""Simple connectivity check for Azure OpenAI (responses API).

Run with: python test/azure_llm_connect/endpoint.py
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict

try:
	from openai import AzureOpenAI  # type: ignore
except Exception as exc:  # pragma: no cover - dependency guard
	print(f"[ERROR] openai SDK is not installed: {exc}")
	sys.exit(1)

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

from config.settings import get_settings  # noqa: E402


def build_client() -> AzureOpenAI:
	settings = get_settings()
	if not settings.aoai_endpoint:
		raise RuntimeError("AOAI_ENDPOINT is not set in .env")
	if not settings.aoai_api_key:
		raise RuntimeError("AOAI_API_KEY is not set in .env")
	api_version = settings.aoai_api_version or "2024-11-20"
	return AzureOpenAI(
		api_key=settings.aoai_api_key,
		azure_endpoint=settings.aoai_endpoint,
		api_version=api_version,
	)


def send_probe(client: AzureOpenAI) -> Dict[str, Any]:
	settings = get_settings()
	deployment = settings.aoai_deployment
	if not deployment:
		raise RuntimeError("AOAI_DEPLOYMENT is not set in .env")
	prompt = "このメッセージを受信したら 'pong' という1単語だけで応答してください。"
	return client.responses.create(
		model=deployment,
		max_output_tokens=32,
		input=[
			{
				"role": "system",
				"content": [
					{
						"type": "input_text",
						"text": "You are a concise assistant.",
					}
				],
			},
			{
				"role": "user",
				"content": [
					{
						"type": "input_text",
						"text": prompt,
					}
				],
			}
		],
	).model_dump()


def main() -> int:
	try:
		client = build_client()
	except Exception as exc:
		print(f"[ERROR] failed to build AzureOpenAI client: {exc}")
		return 1

	try:
		response = send_probe(client)
	except Exception as exc:
		print(f"[ERROR] API call failed: {exc}")
		return 1

	usage = response.get("usage", {})
	output_text = response.get("output_text")
	print("[OK] Azure OpenAI call succeeded")
	if output_text:
		print(f"output_text: {output_text!r}")
	if usage:
		print("usage:")
		print(json.dumps(usage, ensure_ascii=False, indent=2))
	return 0


if __name__ == "__main__":
	raise SystemExit(main())

