#!/usr/bin/env python3
"""
Example usage of get_slider_value_from_lpips_percentage functions.

This script demonstrates how to use the new functions to find slider values
for specific LPIPS percentages.
"""
import numpy as np
from slider_mapping import (
    get_slider_value_from_lpips_percentage,
    get_slider_values_from_lpips_percentages,
)

def example_single_percentage():
    """Example: Get slider value for a single percentage."""
    print("\n" + "="*80)
    print("Example 1: Single Percentage")
    print("="*80)

    # Configuration
    lpips_curve_path = "/data/ru_data/results/trellis_output/validation/test_lpips_curve/obv_toy_0278450d1e324c33904e160e115f6fbd_2/lpips_curve_gradient_9.txt"
    view_images_dir = "path/to/view_images"  # Replace with actual path
    edited_images_dir = "path/to/edited_images"  # Replace with actual path
    editing_prompt_pair = ("make it less X", "make it more X")  # Replace with actual prompts
    out_dir = "path/to/output"  # Replace with actual path

    # Get slider value for 50% LPIPS
    percentage = 50.0

    print(f"\nFinding slider value for {percentage}% LPIPS...")
    print("NOTE: This will generate 3D assets and validate them.")
    print("Make sure all paths are configured correctly!\n")

    # Uncomment to run:
    # slider_value, actual_lpips, is_valid = get_slider_value_from_lpips_percentage(
    #     lpips_curve_path=lpips_curve_path,
    #     percentage=percentage,
    #     view_images_dir=view_images_dir,
    #     edited_images_dir=edited_images_dir,
    #     editing_prompt_pair=editing_prompt_pair,
    #     out_dir=out_dir,
    #     gpu_ids=[0],
    #     max_candidates=5,
    #     invalid_proximity_threshold=0.05,
    # )
    #
    # print(f"\nResult:")
    # print(f"  Slider value: {slider_value:.5f}")
    # print(f"  Actual LPIPS: {actual_lpips:.5f}")
    # print(f"  Valid: {is_valid}")


def example_batch_percentages():
    """Example: Get slider values for multiple percentages with uniqueness."""
    print("\n" + "="*80)
    print("Example 2: Batch Percentages (with uniqueness)")
    print("="*80)

    # Configuration
    lpips_curve_path = "/data/ru_data/results/trellis_output/validation/test_lpips_curve/obv_toy_0278450d1e324c33904e160e115f6fbd_2/lpips_curve_gradient_9.txt"
    view_images_dir = "path/to/view_images"  # Replace with actual path
    edited_images_dir = "path/to/edited_images"  # Replace with actual path
    editing_prompt_pair = ("make it less X", "make it more X")  # Replace with actual prompts
    out_dir = "path/to/output"  # Replace with actual path

    # Get slider values for 9 evenly-spaced percentages
    percentages = np.linspace(0, 100, 9).tolist()

    print(f"\nFinding slider values for {len(percentages)} percentages...")
    print(f"Percentages: {[f'{p:.1f}%' for p in percentages]}")
    print("\nNOTE: This will generate 3D assets and validate them.")
    print("All returned slider values will be unique (differ by at least 0.05).")
    print("Make sure all paths are configured correctly!\n")

    # Uncomment to run:
    # results = get_slider_values_from_lpips_percentages(
    #     lpips_curve_path=lpips_curve_path,
    #     percentages=percentages,
    #     view_images_dir=view_images_dir,
    #     edited_images_dir=edited_images_dir,
    #     editing_prompt_pair=editing_prompt_pair,
    #     out_dir=out_dir,
    #     gpu_ids=[0],
    #     min_slider_distance=0.05,
    #     max_candidates_per_percentage=5,
    #     invalid_proximity_threshold=0.05,
    # )
    #
    # print("\nResults:")
    # for pct, (slider_value, actual_lpips, is_valid) in zip(percentages, results):
    #     status = "✅" if is_valid else "⚠️"
    #     if slider_value is not None:
    #         print(f"{status} {pct:6.2f}% → slider: {slider_value:8.5f}, LPIPS: {actual_lpips:8.5f}")
    #     else:
    #         print(f"❌ {pct:6.2f}% → FAILED")


def example_with_worker_pool():
    """Example: Use GPU worker pool for faster generation."""
    print("\n" + "="*80)
    print("Example 3: Using GPU Worker Pool (Advanced)")
    print("="*80)

    # Configuration
    lpips_curve_path = "/data/ru_data/results/trellis_output/validation/test_lpips_curve/obv_toy_0278450d1e324c33904e160e115f6fbd_2/lpips_curve_gradient_9.txt"
    view_images_dir = "path/to/view_images"  # Replace with actual path
    edited_images_dir = "path/to/edited_images"  # Replace with actual path
    editing_prompt_pair = ("make it less X", "make it more X")  # Replace with actual prompts
    out_dir = "path/to/output"  # Replace with actual path

    percentages = np.linspace(0, 100, 9).tolist()

    print("\nUsing GPU worker pool for faster parallel generation...")
    print("NOTE: Requires multiple GPUs and sufficient VRAM.\n")

    # Uncomment to run:
    # from slider_mapping import GPUWorkerPool
    #
    # # Create worker pool with 2 workers per GPU
    # with GPUWorkerPool(gpu_ids=[0, 1, 2, 3], workers_per_gpu=2) as pool:
    #     results = get_slider_values_from_lpips_percentages(
    #         lpips_curve_path=lpips_curve_path,
    #         percentages=percentages,
    #         view_images_dir=view_images_dir,
    #         edited_images_dir=edited_images_dir,
    #         editing_prompt_pair=editing_prompt_pair,
    #         out_dir=out_dir,
    #         gpu_ids=[0, 1, 2, 3],
    #         pool=pool,  # Pass the worker pool
    #         min_slider_distance=0.05,
    #     )
    #
    #     print("\nResults:")
    #     for pct, (slider_value, actual_lpips, is_valid) in zip(percentages, results):
    #         status = "✅" if is_valid else "⚠️"
    #         print(f"{status} {pct:6.2f}% → {slider_value:8.5f}")


if __name__ == "__main__":
    print("="*80)
    print("LPIPS Percentage Mapping - Usage Examples")
    print("="*80)
    print("\nThese examples show how to use the new functions.")
    print("Uncomment the code in each example to run with your actual data.\n")

    example_single_percentage()
    example_batch_percentages()
    example_with_worker_pool()

    print("\n" + "="*80)
    print("Configuration Steps:")
    print("="*80)
    print("1. Replace 'path/to/...' with your actual paths")
    print("2. Set editing_prompt_pair to your negative/positive prompts")
    print("3. Uncomment the function calls")
    print("4. Run with: conda run -n trellis python example_usage_lpips_percentage.py")
    print("="*80 + "\n")
