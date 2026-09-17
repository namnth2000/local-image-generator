import argparse
import statistics
import time
from pathlib import Path

from optimum.intel import OVLatentConsistencyModelPipeline


MODEL_PATH = Path("models/LCM_Dreamshaper_v7-int8-ov")

PROMPT = (
    "a peaceful Japanese countryside house after rain, "
    "lush green trees, soft natural daylight, cinematic photography, "
    "high detail"
)

SIZE = 384
STEPS_LIST = [2, 3, 4]


def generate(pipe, steps):
    return pipe(
        PROMPT,
        width=SIZE,
        height=SIZE,
        num_inference_steps=steps,
    ).images[0]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device",
        default="GPU",
        choices=["GPU", "CPU"],
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=3,
    )

    args = parser.parse_args()

    print()
    print("LCM OpenVINO - Step Benchmark")
    print("=" * 50)
    print(f"Device:     {args.device}")
    print(f"Resolution: {SIZE}x{SIZE}")
    print(f"Steps:      {STEPS_LIST}")
    print(f"Runs:       {args.runs}")
    print()

    print("Loading model...")

    start = time.perf_counter()

    pipe = OVLatentConsistencyModelPipeline.from_pretrained(
        MODEL_PATH,
        device=args.device,
    )

    load_time = time.perf_counter() - start

    print(f"Load/compile time: {load_time:.2f} s")
    print()

    results = []

    for steps in STEPS_LIST:
        print("=" * 50)
        print(f"{SIZE}x{SIZE} - {steps} steps")
        print("=" * 50)

        # Warm-up riêng cho từng số steps
        print("Warm-up...")

        start = time.perf_counter()
        warmup_image = generate(pipe, steps)
        warmup_time = time.perf_counter() - start

        warmup_image.save(
            f"warmup_{args.device.lower()}_{SIZE}_{steps}steps.png"
        )

        print(f"Warm-up: {warmup_time:.2f} s")
        print()

        times = []

        for run in range(1, args.runs + 1):
            print(f"Run {run}/{args.runs}...", end=" ")

            start = time.perf_counter()

            image = generate(pipe, steps)

            elapsed = time.perf_counter() - start
            times.append(elapsed)

            filename = (
                f"benchmark_{args.device.lower()}_"
                f"{SIZE}_{steps}steps_run{run}.png"
            )

            image.save(filename)

            print(f"{elapsed:.2f} s")

        average = statistics.mean(times)
        median = statistics.median(times)
        minimum = min(times)
        maximum = max(times)

        results.append(
            {
                "steps": steps,
                "average": average,
                "median": median,
                "minimum": minimum,
                "maximum": maximum,
            }
        )

        print()

    print()
    print("FINAL RESULTS")
    print("=" * 65)

    for result in results:
        print(
            f"{SIZE}x{SIZE} - {result['steps']} steps: "
            f"avg {result['average']:.2f}s | "
            f"median {result['median']:.2f}s | "
            f"min {result['minimum']:.2f}s | "
            f"max {result['maximum']:.2f}s"
        )

    print("=" * 65)


if __name__ == "__main__":
    main()
