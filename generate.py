import argparse
import re
import time
from datetime import datetime
from pathlib import Path

from openvino import Core
from optimum.intel import OVLatentConsistencyModelPipeline


DEFAULT_MODEL_PATH = Path("models/LCM_Dreamshaper_v7-int8-ov")
DEFAULT_OUTPUT_DIR = Path("outputs")
DEFAULT_SIZE = 512
DEFAULT_STEPS = 4
DEFAULT_DEVICE = "GPU"


def slugify(text: str, max_length: int = 60) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    if not text:
        text = "image"
    return text[:max_length].strip("-")


def list_devices() -> None:
    core = Core()
    print("Available OpenVINO devices:")
    for device in core.available_devices:
        print(f"- {device}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate images locally with OpenVINO + LCM DreamShaper v7 INT8."
    )

    parser.add_argument(
        "--prompt",
        type=str,
        help="Text prompt to generate an image from.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        choices=["GPU", "CPU"],
        help="Inference device. Default: GPU",
    )

    parser.add_argument(
        "--size",
        type=int,
        default=DEFAULT_SIZE,
        help="Square image size. Recommended values: 256, 384, 512. Default: 384",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help="Number of inference steps. Recommended: 4. Default: 4",
    )

    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the local OpenVINO model folder.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save generated images. Default: outputs",
    )

    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Optional filename prefix.",
    )

    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available OpenVINO devices and exit.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    if not args.prompt:
        parser.error("the following argument is required: --prompt")

    if args.size <= 0:
        parser.error("--size must be a positive integer")

    if args.steps <= 0:
        parser.error("--steps must be a positive integer")

    model_path = args.model_path
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model path not found: {model_path}\n"
            "Make sure you downloaded the model to the correct folder."
        )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("LCM OpenVINO Local Generator")
    print("=" * 50)
    print(f"Device:     {args.device}")
    print(f"Size:       {args.size}x{args.size}")
    print(f"Steps:      {args.steps}")
    print(f"Model path: {model_path}")
    print(f"Output dir: {output_dir}")
    print("=" * 50)
    print()

    print("Loading model...")
    load_start = time.perf_counter()

    pipe = OVLatentConsistencyModelPipeline.from_pretrained(
        model_path,
        device=args.device,
        safety_checker=None,
        requires_safety_checker=False,
    )

    load_elapsed = time.perf_counter() - load_start
    print(f"Model loaded in {load_elapsed:.2f} s")
    print()

    print("Generating image...")
    gen_start = time.perf_counter()

    result = pipe(
        args.prompt,
        width=args.size,
        height=args.size,
        num_inference_steps=args.steps,
    )

    gen_elapsed = time.perf_counter() - gen_start
    image = result.images[0]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prompt_slug = slugify(args.prompt)

    prefix = f"{args.prefix}_" if args.prefix else ""
    filename = (
        f"{prefix}{timestamp}_{args.device.lower()}_"
        f"{args.size}px_{args.steps}steps_{prompt_slug}.png"
    )
    output_path = output_dir / filename

    image.save(output_path)

    print(f"Done in {gen_elapsed:.2f} s")
    print(f"Saved to: {output_path}")
    print()


if __name__ == "__main__":
    main()
