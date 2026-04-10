"""Test sending PDF pages as images to judges via OpenRouter vision API."""

import base64
import json
from pathlib import Path

import pymupdf
from openai import OpenAI

from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, PROMPTS_DIR

client = OpenAI(
    base_url=OPENROUTER_BASE_URL,
    api_key=OPENROUTER_API_KEY,
)

SYSTEM_PROMPT = (PROMPTS_DIR / "system_prompt.txt").read_text()


def pdf_pages_to_base64_images(pdf_path: str, dpi: int = 150) -> list[str]:
    """Convert each page of a PDF to a base64-encoded PNG image."""
    doc = pymupdf.open(pdf_path)
    images = []
    for page in doc:
        # Render page to pixmap
        mat = pymupdf.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        images.append(b64)
    doc.close()
    return images


def test_vision_judge(pdf_path: str, model: str, metric_prompt: str, max_pages: int = 8):
    """Send PDF pages as images to a judge and get a score."""
    print(f"Converting {pdf_path} to images...")
    images = pdf_pages_to_base64_images(pdf_path)[:max_pages]
    print(f"  {len(images)} page images generated")

    # Build multimodal message
    content_parts = [
        {"type": "text", "text": f"METRIC TO EVALUATE:\n\n{metric_prompt}\n\nDOCUMENT PAGES:"},
    ]
    for i, img_b64 in enumerate(images):
        content_parts.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{img_b64}",
            },
        })

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content_parts},
    ]

    print(f"Calling {model}...")
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=16_000,
        temperature=0.3,
    )

    content = response.choices[0].message.content
    usage = response.usage

    print(f"  Tokens — prompt: {usage.prompt_tokens:,}, completion: {usage.completion_tokens:,}, total: {usage.total_tokens:,}")
    print(f"  Estimated cost for this call:")
    # Rough pricing per 1M tokens
    if "sonnet" in model:
        cost = (usage.prompt_tokens * 3 + usage.completion_tokens * 15) / 1_000_000
    elif "gpt" in model:
        cost = (usage.prompt_tokens * 10 + usage.completion_tokens * 30) / 1_000_000
    elif "gemini" in model:
        cost = (usage.prompt_tokens * 2.5 + usage.completion_tokens * 15) / 1_000_000
    else:
        cost = 0
    print(f"    ~${cost:.4f}")

    print(f"\n  Response (first 1000 chars):\n{content[:1000]}")
    return content, usage


if __name__ == "__main__":
    # Test with Grok 4 (8 pages, small) and one metric
    pdf_path = "system_cards/grok_4.pdf"
    metric_prompt = (PROMPTS_DIR / "rubric" / "dangerous_capability_reporting.txt").read_text()

    # Test with Sonnet 4.6 (cheapest)
    test_vision_judge(
        pdf_path=pdf_path,
        model="anthropic/claude-sonnet-4.6",
        metric_prompt=metric_prompt,
    )
