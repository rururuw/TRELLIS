#!/usr/bin/env python3
"""
Test script for get_slider_value_from_lpips_percentage functions.
"""
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from validation.slider_mapping import (
    _parse_lpips_curve,
    _generate_candidates,
    _is_too_close_to_invalid,
    get_slider_value_from_lpips_percentage,
    get_slider_values_from_lpips_percentages,
)

def test_parse_lpips_curve():
    """Test parsing the LPIPS curve file."""
    print("\n" + "="*80)
    print("TEST: Parse LPIPS Curve")
    print("="*80)

    lpips_curve_path = "/data/ru_data/results/trellis_output/validation/test_lpips_curve/obv_toy_0278450d1e324c33904e160e115f6fbd_2/lpips_curve_gradient_9.txt"

    slider_vals, lpips_scores, validities = _parse_lpips_curve(lpips_curve_path)

    print(f"\nTotal observations: {len(slider_vals)}")
    print(f"Valid observations: {np.sum(validities == 1)}")
    print(f"Invalid observations: {np.sum(validities == 0)}")

    valid_mask = (validities == 1)
    print(f"\nValid LPIPS range: [{np.min(lpips_scores[valid_mask]):.5f}, {np.max(lpips_scores[valid_mask]):.5f}]")
    print(f"Valid slider range: [{np.min(slider_vals[valid_mask]):.5f}, {np.max(slider_vals[valid_mask]):.5f}]")

    print("\nFirst 5 observations:")
    for i in range(min(5, len(slider_vals))):
        status = "✅" if validities[i] == 1 else "❌"
        print(f"  {status} Slider: {slider_vals[i]:8.5f}, LPIPS: {lpips_scores[i]:8.5f}")

    return slider_vals, lpips_scores, validities


def test_generate_candidates():
    """Test candidate generation."""
    print("\n" + "="*80)
    print("TEST: Generate Candidates")
    print("="*80)

    lpips_curve_path = "/data/ru_data/results/trellis_output/validation/test_lpips_curve/obv_toy_0278450d1e324c33904e160e115f6fbd_2/lpips_curve_gradient_9.txt"
    slider_vals, lpips_scores, validities = _parse_lpips_curve(lpips_curve_path)

    valid_mask = (validities == 1)
    valid_slider_values = slider_vals[valid_mask]
    valid_lpips_scores = lpips_scores[valid_mask]

    # Test for 50% (middle of range)
    min_lpips = np.min(valid_lpips_scores)
    max_lpips = np.max(valid_lpips_scores)
    target_lpips = min_lpips + (max_lpips - min_lpips) * 0.5

    print(f"\nTarget: 50% → LPIPS = {target_lpips:.5f}")
    print(f"LPIPS range: [{min_lpips:.5f}, {max_lpips:.5f}]")

    candidates = _generate_candidates(target_lpips, valid_slider_values, valid_lpips_scores)

    print(f"\nGenerated {len(candidates)} candidates:")
    for i, c in enumerate(candidates[:10]):  # Show first 10
        print(f"  {i+1}. Slider: {c:.5f}")

    return candidates


def test_proximity_check():
    """Test invalid proximity checking."""
    print("\n" + "="*80)
    print("TEST: Invalid Proximity Check")
    print("="*80)

    lpips_curve_path = "/data/ru_data/results/trellis_output/validation/test_lpips_curve/obv_toy_0278450d1e324c33904e160e115f6fbd_2/lpips_curve_gradient_9.txt"
    slider_vals, lpips_scores, validities = _parse_lpips_curve(lpips_curve_path)

    invalid_mask = (validities == 0)
    invalid_slider_values = slider_vals[invalid_mask]

    print(f"\nInvalid slider values: {invalid_slider_values}")

    test_candidates = [2.50, 2.52, 2.40, -5.0, -4.95, 0.0]
    threshold = 0.05

    print(f"\nTesting candidates with threshold = {threshold}:")
    for candidate in test_candidates:
        is_close = _is_too_close_to_invalid(candidate, invalid_slider_values, threshold)
        status = "❌ TOO CLOSE" if is_close else "✅ SAFE"
        if len(invalid_slider_values) > 0:
            min_dist = np.min(np.abs(invalid_slider_values - candidate))
            print(f"  {status} Candidate: {candidate:6.2f}, min_distance: {min_dist:.3f}")
        else:
            print(f"  {status} Candidate: {candidate:6.2f} (no invalid points)")


def test_single_percentage():
    """Test getting slider value for a single percentage (dry run without actual generation)."""
    print("\n" + "="*80)
    print("TEST: Single Percentage (Dry Run)")
    print("="*80)

    lpips_curve_path = "/data/ru_data/results/trellis_output/validation/test_lpips_curve/obv_toy_0278450d1e324c33904e160e115f6fbd_2/lpips_curve_gradient_9.txt"

    # Parse to show what would happen
    slider_vals, lpips_scores, validities = _parse_lpips_curve(lpips_curve_path)
    valid_mask = (validities == 1)
    valid_slider_values = slider_vals[valid_mask]
    valid_lpips_scores = lpips_scores[valid_mask]

    percentage = 50.0
    min_lpips = np.min(valid_lpips_scores)
    max_lpips = np.max(valid_lpips_scores)
    target_lpips = min_lpips + (max_lpips - min_lpips) * (percentage / 100.0)

    print(f"\nPercentage: {percentage}%")
    print(f"Target LPIPS: {target_lpips:.5f}")

    candidates = _generate_candidates(target_lpips, valid_slider_values, valid_lpips_scores)
    print(f"\nTop 5 candidates that would be tested:")
    for i, c in enumerate(candidates[:5]):
        # Find nearest valid point to estimate LPIPS
        idx = np.argmin(np.abs(valid_slider_values - c))
        est_lpips = valid_lpips_scores[idx]
        print(f"  {i+1}. Slider: {c:8.5f}, Est. LPIPS: {est_lpips:8.5f}, Diff: {abs(est_lpips - target_lpips):8.5f}")

    print("\n⚠️ Actual generation and validation skipped (dry run)")


def test_batch_percentages():
    """Test getting slider values for multiple percentages (dry run)."""
    print("\n" + "="*80)
    print("TEST: Batch Percentages (Dry Run)")
    print("="*80)

    lpips_curve_path = "/data/ru_data/results/trellis_output/validation/test_lpips_curve/obv_toy_0278450d1e324c33904e160e115f6fbd_2/lpips_curve_gradient_9.txt"

    percentages = np.linspace(0, 100, 9).tolist()
    print(f"\nPercentages to process: {[f'{p:.1f}%' for p in percentages]}")

    # Parse curve
    slider_vals, lpips_scores, validities = _parse_lpips_curve(lpips_curve_path)
    valid_mask = (validities == 1)
    valid_slider_values = slider_vals[valid_mask]
    valid_lpips_scores = lpips_scores[valid_mask]

    min_lpips = np.min(valid_lpips_scores)
    max_lpips = np.max(valid_lpips_scores)

    print(f"\nLPIPS range: [{min_lpips:.5f}, {max_lpips:.5f}]")
    print(f"\nPredicted targets (before validation):")

    for pct in percentages:
        target_lpips = min_lpips + (max_lpips - min_lpips) * (pct / 100.0)
        candidates = _generate_candidates(target_lpips, valid_slider_values, valid_lpips_scores)

        if len(candidates) > 0:
            best_candidate = candidates[0]
            idx = np.argmin(np.abs(valid_slider_values - best_candidate))
            est_lpips = valid_lpips_scores[idx]
            print(f"  {pct:6.2f}% → Target LPIPS: {target_lpips:8.5f}, Best candidate: {best_candidate:8.5f}")

    print("\n⚠️ Actual generation and validation skipped (dry run)")
    print("⚠️ Uniqueness constraint would be applied during actual validation")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("LPIPS Percentage Mapping - Test Suite")
    print("="*80)

    try:
        # Run tests
        test_parse_lpips_curve()
        test_generate_candidates()
        test_proximity_check()
        test_single_percentage()
        test_batch_percentages()

        print("\n" + "="*80)
        print("✅ All tests completed successfully!")
        print("="*80)
        print("\nTo run actual generation and validation, call:")
        print("  get_slider_value_from_lpips_percentage(...)")
        print("  get_slider_values_from_lpips_percentages(...)")
        print("\nwith appropriate view_images_dir, edited_images_dir, etc.")
        print("="*80 + "\n")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
