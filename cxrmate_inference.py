"""
CXRMate-2 Inference Utility
============================
Supports single-image and batch (directory) inference for CXRMate-2.

Usage:
    python cxrmate_inference.py <image_or_directory>
    python cxrmate_inference.py images/test_xray.jpg
    python cxrmate_inference.py images/

Research use only. AI-generated reports must not be used as a standalone
clinical diagnosis.
"""

import argparse
import csv
import json
import sys
import time
import traceback
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    GenerationConfig,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_ID = "aehrc/cxrmate-2"

SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp",
}

RESULTS_DIR = Path(__file__).resolve().parent / "results"

DISCLAIMER = (
    "Research use only. AI-generated reports must not be used "
    "as a standalone clinical diagnosis."
)

SEPARATOR = "=" * 60


# ---------------------------------------------------------------------------
# Model / Processor loading
# ---------------------------------------------------------------------------

def load_processor():
    """Load the CXRMate-2 processor (tokenizer + image processor)."""
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        local_files_only=True,
    )
    print("Processor loaded.")
    return processor


def load_model(device: torch.device):
    """Load the CXRMate-2 model in FP16 and move to *device*."""
    print("Loading FP16 model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        dtype=torch.float16,
        local_files_only=True,
    )
    model = model.to(device)
    model.eval()
    print("Model loaded.")
    print("Model dtype:", next(model.parameters()).dtype)
    return model


def load_generation_config():
    """Load the generation configuration shipped with the model."""
    return GenerationConfig.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        local_files_only=True,
    )


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_image(path: Path) -> None:
    """Raise if *path* is not a supported, existing image file."""
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    if not path.is_file():
        raise ValueError(f"Not a file: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported image extension '{path.suffix}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )


def collect_images(directory: Path) -> list[Path]:
    """Return sorted list of supported image files inside *directory*."""
    images = [
        p for p in sorted(directory.iterdir())
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return images


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(
    image_path: Path,
    processor,
    model,
    generation_config,
    device: torch.device,
) -> dict:
    """
    Run inference on a single image and return a result dict.

    Returns
    -------
    dict with keys: image, image_path, status, inference_time_seconds,
                    findings, impression, error
    """
    result = {
        "image": image_path.name,
        "image_path": str(image_path),
        "status": "failed",
        "inference_time_seconds": 0.0,
        "findings": "",
        "impression": "",
        "error": "",
    }

    try:
        validate_image(image_path)

        # Process the image through the processor
        processed = processor(images=str(image_path))
        processed = processed.to(device=device)

        # CXRMate-2 creates time_deltas as FP32 during generation,
        # while the model is loaded in FP16 — cast to match.
        if "time_deltas" in processed:
            processed["time_deltas"] = processed["time_deltas"].to(
                dtype=torch.float16,
                device=device,
            )

        # Generate
        start = time.perf_counter()
        with torch.no_grad():
            generated_ids = model.generate(
                **processed,
                generation_config=generation_config,
            )
        elapsed = time.perf_counter() - start

        # Decode
        findings, impression = processor.split_and_decode_sections(
            generated_ids
        )

        # Normalise list outputs to strings
        if isinstance(findings, list):
            findings = "\n".join(findings)
        if isinstance(impression, list):
            impression = "\n".join(impression)

        result.update({
            "status": "success",
            "inference_time_seconds": round(elapsed, 2),
            "findings": findings,
            "impression": impression,
        })

    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["inference_time_seconds"] = 0.0

    return result


# ---------------------------------------------------------------------------
# Result persistence
# ---------------------------------------------------------------------------

def save_json_result(result: dict) -> Path:
    """Save a per-image JSON result file and return its path."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = Path(result["image"]).stem
    out_path = RESULTS_DIR / f"{stem}.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)
    return out_path


def save_summary_csv(results: list[dict]) -> Path:
    """Write results/summary.csv with one row per image."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "summary.csv"
    fieldnames = [
        "image", "status", "inference_time_seconds",
        "findings", "impression", "error",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, "") for k in fieldnames})
    return csv_path


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def print_result(result: dict) -> None:
    """Pretty-print a single inference result to the console."""
    print()
    print(SEPARATOR)
    print("FINDINGS")
    print(SEPARATOR)
    print(result["findings"] if result["findings"] else "(empty)")
    print()
    print(SEPARATOR)
    print("IMPRESSION")
    print(SEPARATOR)
    print(result["impression"] if result["impression"] else "(empty)")
    print()
    print(f"Inference time: {result['inference_time_seconds']:.2f} seconds")


def print_batch_summary(results: list[dict], total_time: float) -> None:
    """Print a summary table after batch inference."""
    total = len(results)
    success = sum(1 for r in results if r["status"] == "success")
    failed = total - success
    avg = total_time / total if total else 0.0

    print()
    print(SEPARATOR)
    print("BATCH SUMMARY")
    print(SEPARATOR)
    print(f"Total images:          {total}")
    print(f"Successful:            {success}")
    print(f"Failed:                {failed}")
    print(f"Total time:            {total_time:.2f} seconds")
    print(f"Average time/image:    {avg:.2f} seconds")

    if failed:
        print()
        print("Failed images:")
        for r in results:
            if r["status"] != "success":
                print(f"  - {r['image']}: {r['error']}")


# ---------------------------------------------------------------------------
# Processing modes
# ---------------------------------------------------------------------------

def process_single_image(
    image_path: Path,
    processor,
    model,
    generation_config,
    device: torch.device,
) -> dict:
    """Process one image, print output, save JSON, return result."""
    print()
    print("Preparing image...")
    print(f"Image: {image_path}")
    print()
    print("Starting generation...")
    print("This may take a while on CPU.")

    result = run_inference(
        image_path, processor, model, generation_config, device,
    )

    if result["status"] == "success":
        print_result(result)
        json_path = save_json_result(result)
        print(f"Result saved to: {json_path}")
    else:
        print()
        print(f"ERROR: {result['error']}")
        json_path = save_json_result(result)
        print(f"Error details saved to: {json_path}")

    return result


def process_directory(
    directory: Path,
    processor,
    model,
    generation_config,
    device: torch.device,
) -> list[dict]:
    """Process all supported images in *directory*."""
    images = collect_images(directory)
    if not images:
        print(f"No supported images found in: {directory}")
        print(f"Supported extensions: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
        return []

    print(f"Found {len(images)} image(s) in: {directory}")
    print()

    results = []
    batch_start = time.perf_counter()

    for idx, img_path in enumerate(images, 1):
        print(SEPARATOR)
        print(f"[{idx}/{len(images)}] Processing: {img_path.name}")
        print(SEPARATOR)

        result = run_inference(
            img_path, processor, model, generation_config, device,
        )
        results.append(result)

        if result["status"] == "success":
            print_result(result)
            json_path = save_json_result(result)
            print(f"Result saved to: {json_path}")
        else:
            print(f"  FAILED: {result['error']}")
            json_path = save_json_result(result)
            print(f"  Error details saved to: {json_path}")

        print()

    batch_elapsed = time.perf_counter() - batch_start

    # Save CSV summary
    csv_path = save_summary_csv(results)
    print(f"Summary CSV saved to: {csv_path}")

    print_batch_summary(results, batch_elapsed)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CXRMate-2 Inference — single image or batch directory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=DISCLAIMER,
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to a single image file or a directory of images.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    input_path = Path(args.input).resolve()

    print(SEPARATOR)
    print("CXRMate-2 Inference")
    print(SEPARATOR)
    print(f"Input: {input_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print()
    print(DISCLAIMER)
    print()

    # ------------------------------------------------------------------
    # Load model stack ONCE
    # ------------------------------------------------------------------
    try:
        processor = load_processor()
        model = load_model(device)
        generation_config = load_generation_config()
    except Exception as exc:
        print(f"\nFATAL: Failed to load model/processor: {exc}")
        traceback.print_exc()
        sys.exit(1)

    print()

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------
    if input_path.is_dir():
        process_directory(
            input_path, processor, model, generation_config, device,
        )
    elif input_path.is_file():
        result = process_single_image(
            input_path, processor, model, generation_config, device,
        )
        if result["status"] != "success":
            sys.exit(1)
    else:
        print(f"ERROR: Path does not exist: {input_path}")
        sys.exit(1)

    print()
    print(SEPARATOR)
    print("DONE")
    print(SEPARATOR)


if __name__ == "__main__":
    main()