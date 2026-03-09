import sys
sys.path.append("../")
from gen_image_pair_prompts import get_qwen_lora_pipeline, generate_image_qwen_lora
from trellis.pipelines import TrellisAttributeSliderPipeline, TrellisImageTo3DAttributeSliderPipeline
import utils3d
import json
from subprocess import DEVNULL, call
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import cpu_count
import os
import numpy as np
import base64
from io import BytesIO
sys.path.append(os.path.join(os.path.dirname(__file__), '../dataset_toolkits'))
from utils import sphere_hammersley_sequence
import csv

# Flask imports for server functionality
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Lazy imports for heavy modules (only load when needed)
_qwen_pipeline = None
_trellis_pipeline = None
_trellis_img_pipeline = None

def get_qwen_pipeline():
    """Lazy load Qwen LoRA pipeline."""
    global _qwen_pipeline
    if _qwen_pipeline is None:
        print("Loading Qwen LoRA pipeline...")
        from gen_image_pair_prompts import get_qwen_lora_pipeline
        _qwen_pipeline = get_qwen_lora_pipeline()
        print("Qwen LoRA pipeline loaded.")
    return _qwen_pipeline

def get_trellis_text_pipeline():
    """Lazy load TRELLIS text-to-3D pipeline."""
    global _trellis_pipeline
    if _trellis_pipeline is None:
        print("Loading TRELLIS text-to-3D pipeline...")
        from trellis.pipelines import TrellisAttributeSliderPipeline
        _trellis_pipeline = TrellisAttributeSliderPipeline.from_pretrained("microsoft/TRELLIS-text-large")
        _trellis_pipeline.cuda()
        print("TRELLIS text-to-3D pipeline loaded.")
    return _trellis_pipeline

def get_trellis_img_pipeline():
    """Lazy load TRELLIS image-to-3D pipeline."""
    global _trellis_img_pipeline
    if _trellis_img_pipeline is None:
        print("Loading TRELLIS image-to-3D pipeline...")
        from trellis.pipelines import TrellisImageTo3DAttributeSliderPipeline
        _trellis_img_pipeline = TrellisImageTo3DAttributeSliderPipeline.from_pretrained("microsoft/TRELLIS-image-large")
        _trellis_img_pipeline.cuda()
        print("TRELLIS image-to-3D pipeline loaded.")
    return _trellis_img_pipeline

def _render_views_worker(args: tuple) -> dict:
    """
    Worker function: renders a subset of views for an asset.
    Args: (asset_path, views_subset, resolution, temp_out_dir, worker_id, save_mesh)
    """
    asset_path, views_subset, resolution, temp_out_dir, worker_id, save_mesh = args
    
    BLENDER_PATH = 'blender'
    RENDER_SCRIPT = os.path.join(os.path.dirname(__file__), '../dataset_toolkits', 'blender_script', 'render.py')
    
    cmd = [
        BLENDER_PATH, '-b', '-P', RENDER_SCRIPT,
        '--',
        '--views', json.dumps(views_subset),
        '--object', os.path.abspath(asset_path),
        '--resolution', str(resolution),
        '--output_folder', temp_out_dir,
        '--engine', 'CYCLES',
    ]
    if save_mesh:
        cmd.append('--save_mesh')
    
    try:
        ret = call(cmd, stdout=DEVNULL, stderr=DEVNULL)
        if ret != 0:
            return {'worker_id': worker_id, 'status': 'failed', 'views': len(views_subset)}
        return {'worker_id': worker_id, 'status': 'success', 'views': len(views_subset), 'temp_dir': temp_out_dir}
    except Exception as e:
        return {'worker_id': worker_id, 'status': 'error', 'message': str(e)}


def get_multi_view_from_3d_assets_mp(
    asset_path: str, 
    num_views: int, 
    out_dir: str, 
    max_workers: int = None,
    random_seed: int = None
) -> bool:
    """
    Multiprocess version: Renders multiple views of a single 3D asset in parallel.
    
    Splits the views among multiple Blender processes to speed up rendering.
    
    Args:
        asset_path: Path to the 3D asset file (GLB, OBJ, etc.)
        num_views: Total number of views to render
        out_dir: Output directory for rendered images
        max_workers: Number of parallel Blender processes (default: min(num_views, cpu_count-2))
        random_seed: Random seed for camera pose generation (default: None for random)
    
    Returns:
        True if successful, False otherwise
    """
    RESOLUTION = 512
    
    if max_workers is None:
        max_workers = max(1, min(num_views, cpu_count() - 2))
    
    # Ensure we don't have more workers than views
    max_workers = min(max_workers, num_views)
    
    os.makedirs(out_dir, exist_ok=True)
    
    # Generate all camera poses upfront with optional seeding
    if random_seed is not None:
        rng = np.random.default_rng(random_seed)
        offset = (rng.random(), rng.random())
    else:
        offset = (np.random.rand(), np.random.rand())
    
    all_views = []
    for i in range(num_views):
        y, p = sphere_hammersley_sequence(i, num_views, offset)
        all_views.append({'yaw': y, 'pitch': p, 'radius': 2.0, 'fov': 40 / 180 * np.pi})
    
    # Split views among workers
    views_per_worker = []
    chunk_size = num_views // max_workers
    remainder = num_views % max_workers
    
    start_idx = 0
    for i in range(max_workers):
        # Distribute remainder among first workers
        end_idx = start_idx + chunk_size + (1 if i < remainder else 0)
        views_per_worker.append(all_views[start_idx:end_idx])
        start_idx = end_idx
    
    # Create temp directories for each worker
    temp_dirs = []
    worker_args = []
    for i, views_chunk in enumerate(views_per_worker):
        temp_dir = os.path.join(out_dir, f'_temp_worker_{i}')
        os.makedirs(temp_dir, exist_ok=True)
        temp_dirs.append(temp_dir)
        # Only first worker saves the mesh to avoid redundant work
        save_mesh = (i == 0)
        worker_args.append((asset_path, views_chunk, RESOLUTION, temp_dir, i, save_mesh))
    
    print(f"Rendering {num_views} views with {max_workers} parallel Blender processes...")
    
    # Launch parallel rendering
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for result in executor.map(_render_views_worker, worker_args):
            status = result['status']
            if status == 'success':
                print(f"  Worker {result['worker_id']}: rendered {result['views']} views")
            else:
                print(f"  Worker {result['worker_id']}: {status} - {result.get('message', '')}")
            results.append(result)
    
    # Check if all workers succeeded
    if not all(r['status'] == 'success' for r in results):
        print(f"Failed to render some views for {asset_path}")
        return False
    
    # Merge results: combine transforms.json and move images
    print("Merging results...")
    combined_frames = []
    view_counter = 0
    
    for i, temp_dir in enumerate(temp_dirs):
        transforms_path = os.path.join(temp_dir, 'transforms.json')
        if os.path.exists(transforms_path):
            with open(transforms_path, 'r') as f:
                data = json.load(f)
            
            # Rename and move images to final directory with sequential numbering
            for frame in data['frames']:
                old_path = os.path.join(temp_dir, frame['file_path'])
                new_filename = f'{view_counter:03d}.png'
                new_path = os.path.join(out_dir, new_filename)
                
                if os.path.exists(old_path):
                    os.rename(old_path, new_path)
                
                # Update frame info
                frame['file_path'] = new_filename
                combined_frames.append(frame)
                view_counter += 1
            
            # Copy mesh from first worker
            if i == 0:
                mesh_src = os.path.join(temp_dir, 'mesh.ply')
                mesh_dst = os.path.join(out_dir, 'mesh.ply')
                if os.path.exists(mesh_src):
                    os.rename(mesh_src, mesh_dst)
    
    # Write combined transforms.json
    # Use camera parameters from first worker's data (they should all be same except frames)
    with open(os.path.join(temp_dirs[0], 'transforms.json'), 'r') as f:
        base_data = json.load(f)
    
    base_data['frames'] = combined_frames
    with open(os.path.join(out_dir, 'transforms.json'), 'w') as f:
        json.dump(base_data, f, indent=2)
    
    # Cleanup temp directories
    import shutil
    for temp_dir in temp_dirs:
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    print(f"Successfully rendered {num_views} views from {asset_path} using {max_workers} workers.")
    return True


# ==============================================================================
# Flask Server Setup
# ==============================================================================

# Setup paths
DATA_DIR = '/data/ru_data/user_logs'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(DATA_DIR, exist_ok=True)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

@app.route('/')
def index():
    return jsonify({
        'service': 'TRELLIS Server',
        'status': 'running',
        'endpoints': [
            '/health',
            '/generate_image',
            '/text_to_3d',
            '/image_to_3d',
            '/render_views',
            '/outputs/<path:filename>'
        ]
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'ok', 'service': 'trellis'})


@app.route('/render_views', methods=['POST'])
def render_views():
    """Render multiple views of a 3D asset."""
    try:
        # Handle file upload
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        num_views = int(request.form.get('num_views', 25))
        max_workers = int(request.form.get('max_workers', 4))
        
        # Save uploaded file temporarily
        filename = file.filename
        asset_name = os.path.splitext(filename)[0]
        temp_path = os.path.join(DATA_DIR, 'temp_assets', filename)
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        file.save(temp_path)
        
        # Output directory
        out_dir = os.path.join(DATA_DIR, f'{asset_name}_views')
        
        # Render views
        success = get_multi_view_from_3d_assets_mp(
            temp_path, num_views, out_dir, max_workers
        )
        
        if not success:
            return jsonify({'error': 'Rendering failed'}), 500
        
        # Get list of rendered images
        rendered_images = []
        for f in sorted(os.listdir(out_dir)):
            if f.endswith('.png'):
                rendered_images.append(f'/outputs/{asset_name}_views/{f}')
        
        return jsonify({
            'success': True,
            'output_dir': f'{asset_name}_views',
            'num_views': len(rendered_images),
            'images': rendered_images
        })
        
    except Exception as e:
        print(f"Render views error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


def run_server(host='0.0.0.0', port=5002, debug=True):
    """Run the Flask server."""
    print(f"Starting TRELLIS server on {host}:{port}")
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='TRELLIS Server and Batch Processing')
    parser.add_argument('--port', type=int, default=5002, help='Server port (default: 5002)')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Server host (default: 0.0.0.0)')
    
    args = parser.parse_args()
    

    # Run as Flask server
    run_server(host=args.host, port=args.port, debug=True)