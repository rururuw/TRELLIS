# LPIPS Percentage Mapping Functions

This document describes the new functions added to `slider_mapping.py` for finding slider values based on LPIPS percentages.

## Overview

Given an LPIPS curve file and a percentage (0-100%), these functions find the best slider value that produces the corresponding LPIPS score, with quality validation.

## Functions

### 1. `get_slider_value_from_lpips_percentage`

Find the best valid slider value for a **single** LPIPS percentage.

**Signature:**
```python
def get_slider_value_from_lpips_percentage(
    lpips_curve_path: str,
    percentage: float,  # 0-100
    view_images_dir: str,
    edited_images_dir: str,
    editing_prompt_pair: Tuple[str, str],  # (neg_prompt, pos_prompt)
    out_dir: str,
    gpu_ids: List[int] = [0],
    pipelines: Dict[int, object] = None,
    conds: Dict[int, dict] = None,
    pool: GPUWorkerPool = None,
    max_candidates: int = 5,
    invalid_proximity_threshold: float = 0.05,
) -> Tuple[float, float, bool]:
    """
    Returns: (slider_value, actual_lpips_score, is_valid)
    """
```

**Example:**
```python
slider_value, actual_lpips, is_valid = get_slider_value_from_lpips_percentage(
    lpips_curve_path="path/to/lpips_curve_gradient_9.txt",
    percentage=50.0,  # Find slider for middle of LPIPS range
    view_images_dir="path/to/views",
    edited_images_dir="path/to/edited_views",
    editing_prompt_pair=("less shiny", "more shiny"),
    out_dir="path/to/output",
    gpu_ids=[0],
)
print(f"50% → slider: {slider_value:.5f}, LPIPS: {actual_lpips:.5f}, valid: {is_valid}")
```

### 2. `get_slider_values_from_lpips_percentages`

Find valid slider values for **multiple** LPIPS percentages with **uniqueness guarantee**.

**Signature:**
```python
def get_slider_values_from_lpips_percentages(
    lpips_curve_path: str,
    percentages: List[float],  # List of 0-100 values
    view_images_dir: str,
    edited_images_dir: str,
    editing_prompt_pair: Tuple[str, str],
    out_dir: str,
    gpu_ids: List[int] = [0],
    pipelines: Dict[int, object] = None,
    conds: Dict[int, dict] = None,
    pool: GPUWorkerPool = None,
    min_slider_distance: float = 0.05,  # Minimum distance between returned values
    max_candidates_per_percentage: int = 5,
    invalid_proximity_threshold: float = 0.05,
) -> List[Tuple[float, float, bool]]:
    """
    Returns: List of (slider_value, actual_lpips_score, is_valid) for each percentage.
    All slider values are guaranteed unique (differ by at least min_slider_distance).
    """
```

**Example:**
```python
import numpy as np

percentages = np.linspace(0, 100, 9).tolist()  # [0, 12.5, 25, ..., 100]

results = get_slider_values_from_lpips_percentages(
    lpips_curve_path="path/to/lpips_curve_gradient_9.txt",
    percentages=percentages,
    view_images_dir="path/to/views",
    edited_images_dir="path/to/edited_views",
    editing_prompt_pair=("less shiny", "more shiny"),
    out_dir="path/to/output",
    gpu_ids=[0],
    min_slider_distance=0.05,  # Ensure all values differ by at least 0.05
)

for pct, (slider, lpips, valid) in zip(percentages, results):
    print(f"{pct:6.2f}% → slider: {slider:8.5f}, LPIPS: {lpips:8.5f}, valid: {valid}")
```

## How It Works

### Step 1: Parse LPIPS Curve
- Reads the curve file (format: `slider_value,lpips_score,validity`)
- Separates valid (validity=1) and invalid (validity=0) observations
- Invalid points are used as warnings, not hard boundaries

### Step 2: Calculate Target LPIPS
```python
min_lpips = min(valid_lpips_scores)
max_lpips = max(valid_lpips_scores)
target_lpips = min_lpips + (max_lpips - min_lpips) * (percentage / 100.0)
```

### Step 3: Generate Candidates
Uses three strategies:
1. **Direct lookup**: Find valid observations with closest LPIPS scores
2. **Interpolation**: Interpolate along the valid LPIPS curve
3. **Neighbors**: Add nearby slider values around interpolated estimate

### Step 4: Filter Candidates
- **Invalid proximity check**: Deprioritize candidates within 0.05 of invalid observations
- **Uniqueness check** (batch only): Filter out values too close to already-assigned sliders

### Step 5: Validate Each Candidate
For each candidate (in order of estimated accuracy):
1. Generate 3D asset if not cached
2. Compute actual LPIPS score
3. Verify quality using `get_quality_change_verification`:
   - VLM-based quality check
   - Compare with reference frames

### Step 6: Return Result
- If validation passes → return that slider value
- If all fail → return closest valid observation from curve (with `is_valid=False`)

## LPIPS Curve File Format

The curve file should have two parts separated by `\n\n`:

**Part 1 (required):** Observations
```
slider_value,lpips_score,validity
-5.00000,-0.11290,0
-4.85986,-0.11256,1
-4.75000,-0.10609,1
...
5.00000,0.23031,1
```

**Part 2 (optional):** Metadata (ignored by these functions)
```

Time taken: 486.21 seconds
Valid points: 19, Banned points: 9
```

## Key Features

### 1. Smart Candidate Selection
- Combines direct lookup + interpolation for robustness
- Handles non-monotonic LPIPS curves
- Avoids candidates too close to invalid observations

### 2. Quality Validation
- Uses same validation as `get_lpips_curve_gradient_based`
- VLM-based quality control
- LPIPS score verification

### 3. Uniqueness Guarantee (Batch Mode)
- All returned slider values differ by at least `min_slider_distance`
- Automatic conflict resolution via perturbation
- Graceful fallback to valid observations

### 4. Efficient Caching
- Reuses existing generated frames
- Parses curve only once in batch mode
- Optional GPU worker pool for parallelization

## Performance Tips

### For Single Percentage
```python
# Pre-load models if calling multiple times
pipelines = load_pipelines(gpu_ids=[0])
conds = load_conds(pipelines, view_images_dir, edited_images_dir)

slider, lpips, valid = get_slider_value_from_lpips_percentage(
    ...,
    pipelines=pipelines,
    conds=conds,
)
```

### For Batch Processing
```python
# Use GPU worker pool for parallel generation
from slider_mapping import GPUWorkerPool

with GPUWorkerPool(gpu_ids=[0, 1, 2, 3], workers_per_gpu=2) as pool:
    results = get_slider_values_from_lpips_percentages(
        ...,
        pool=pool,
    )
```

## Parameters

### Common Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lpips_curve_path` | str | - | Path to LPIPS curve file |
| `percentage(s)` | float / List[float] | - | Target percentage(s) 0-100 |
| `view_images_dir` | str | - | Original multi-view images |
| `edited_images_dir` | str | - | Edited view images (pos/neg) |
| `editing_prompt_pair` | Tuple[str, str] | - | (negative_prompt, positive_prompt) |
| `out_dir` | str | - | Output directory for generated frames |
| `gpu_ids` | List[int] | [0] | GPU device IDs |
| `pipelines` | Dict | None | Pre-loaded pipelines (optional) |
| `conds` | Dict | None | Pre-loaded conditions (optional) |
| `pool` | GPUWorkerPool | None | Worker pool (optional) |
| `invalid_proximity_threshold` | float | 0.05 | Avoid candidates within this distance of invalid points |

### Batch-Specific Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_slider_distance` | float | 0.05 | Minimum distance between returned slider values |
| `max_candidates_per_percentage` | int | 5 | Max candidates to validate per percentage |

## Return Values

### Single Percentage
Returns `Tuple[float, float, bool]`:
- `slider_value`: Best slider value found
- `actual_lpips_score`: Actual LPIPS score (from curve or newly computed)
- `is_valid`: Whether validation passed (`False` if using fallback)

### Batch Percentages
Returns `List[Tuple[float, float, bool]]`:
- One tuple per input percentage, in same order
- All `slider_value`s are guaranteed unique
- If no unique value available: `(None, None, False)`

## Testing

Run tests with:
```bash
conda run -n trellis python -c "
import sys
sys.path.insert(0, '/home/rwang/TRELLIS')
from slider_mapping import _parse_lpips_curve, _generate_candidates
import numpy as np

# Your test code here
"
```

See `example_usage_lpips_percentage.py` for complete examples.

## Related Functions

These functions build on:
- `get_lpips_curve_gradient_based`: Generates the LPIPS curve
- `get_quality_change_verification`: VLM-based quality validation
- `get_LPIPS_score_relative_to_reference`: LPIPS computation
- `run_generation_multiprocess`: Parallel frame generation
