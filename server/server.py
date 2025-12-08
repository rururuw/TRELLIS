import os
import torch
os.environ['SPCONV_ALGO'] = 'native'

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List
import imageio
import numpy as np
import open3d as o3d
import uuid
import json
from datetime import datetime
from pathlib import Path

from trellis.pipelines import TrellisTextTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils

app = FastAPI(title="TRELLIS 3D Generation API")

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
pipeline = None
current_base_mesh = None
outputs_dir = Path("server_outputs")
outputs_dir.mkdir(exist_ok=True)

# Store job status
jobs = {}


class TextTo3DRequest(BaseModel):
    prompt: str
    seed: Optional[int] = 1
    sparse_steps: Optional[int] = 12
    sparse_cfg: Optional[float] = 7.5
    slat_steps: Optional[int] = 12
    slat_cfg: Optional[float] = 7.5


class VariantRequest(BaseModel):
    prompt: str
    seed: Optional[int] = 1
    steps: Optional[int] = 12
    cfg_strength: Optional[float] = 7.5


class VariantBatchRequest(BaseModel):
    prompt: str
    seed: Optional[int] = 1
    steps: Optional[int] = 12
    cfg_strengths: List[float] = [-10, -5, -2.5, 0, 2.5, 5, 10]


@app.on_event("startup")
async def startup_event():
    """Load the pipeline on startup"""
    global pipeline
    print("Loading TRELLIS pipeline...")
    pipeline = TrellisTextTo3DPipeline.from_pretrained("microsoft/TRELLIS-text-large")
    pipeline.cuda()
    print("Pipeline loaded successfully!")


@app.get("/")
async def root():
    return {"message": "TRELLIS 3D Generation API", "status": "running"}


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "pipeline_loaded": pipeline is not None,
        "base_mesh_loaded": current_base_mesh is not None
    }


@app.post("/generate")
async def generate_text_to_3d(request: TextTo3DRequest, background_tasks: BackgroundTasks):
    """Generate a 3D model from text prompt"""
    global current_base_mesh
    
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded")
    
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "processing",
        "type": "generation",
        "prompt": request.prompt,
        "created_at": datetime.now().isoformat()
    }
    
    try:
        print(f"Generating 3D model for: {request.prompt}")
        
        # Run the pipeline
        outputs = pipeline.run(
            request.prompt,
            seed=request.seed,
            sparse_structure_sampler_params={
                "steps": request.sparse_steps,
                "cfg_strength": request.sparse_cfg,
            },
            slat_sampler_params={
                "steps": request.slat_steps,
                "cfg_strength": request.slat_cfg,
            },
        )
        
        # Extract and save the base mesh
        mesh_result = outputs['mesh'][0]
        vertices_np = mesh_result.vertices.cpu().numpy()
        faces_np = mesh_result.faces.cpu().numpy()
        
        base_mesh = o3d.geometry.TriangleMesh()
        base_mesh.vertices = o3d.utility.Vector3dVector(vertices_np.astype(np.float64))
        base_mesh.triangles = o3d.utility.Vector3iVector(faces_np.astype(np.int32))
        
        # Save mesh to file
        job_dir = outputs_dir / job_id
        job_dir.mkdir(exist_ok=True)
        
        mesh_path = job_dir / "base_mesh.ply"
        o3d.io.write_triangle_mesh(str(mesh_path), base_mesh)
        
        # Reload mesh (workaround for Open3D bugs)
        current_base_mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        
        # Render video
        video_gs = render_utils.render_video(outputs['gaussian'][0])['color']
        video_mesh = render_utils.render_video(outputs['mesh'][0])['normal']
        video = [np.concatenate([frame_gs, frame_mesh], axis=1) for frame_gs, frame_mesh in zip(video_gs, video_mesh)]
        
        video_path = job_dir / "output.mp4"
        imageio.mimsave(str(video_path), video, fps=30)
        
        jobs[job_id] = {
            "status": "completed",
            "type": "generation",
            "prompt": request.prompt,
            "created_at": jobs[job_id]["created_at"],
            "completed_at": datetime.now().isoformat(),
            "video_url": f"/outputs/{job_id}/output.mp4",
            "mesh_url": f"/outputs/{job_id}/base_mesh.ply",
            "vertices_count": len(base_mesh.vertices),
            "faces_count": len(base_mesh.triangles)
        }
        
        return jobs[job_id]
        
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/variant")
async def generate_variant(request: VariantRequest):
    """Generate a variant of the current base mesh"""
    global current_base_mesh
    
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded")
    
    if current_base_mesh is None:
        raise HTTPException(status_code=400, detail="No base mesh loaded. Generate a 3D model first.")
    
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "processing",
        "type": "variant",
        "prompt": request.prompt,
        "cfg_strength": request.cfg_strength,
        "created_at": datetime.now().isoformat()
    }
    
    try:
        print(f"Generating variant with cfg_strength={request.cfg_strength}")
        
        # Run variant generation
        outputs = pipeline.run_variant(
            current_base_mesh,
            request.prompt,
            seed=request.seed,
            slat_sampler_params={
                "steps": request.steps,
                "cfg_strength": request.cfg_strength,
            },
        )
        
        # Create job directory
        job_dir = outputs_dir / job_id
        job_dir.mkdir(exist_ok=True)
        
        # Render video
        video_gs = render_utils.render_video(outputs['gaussian'][0])['color']
        video_mesh = render_utils.render_video(outputs['mesh'][0])['normal']
        video = [np.concatenate([frame_gs, frame_mesh], axis=1) for frame_gs, frame_mesh in zip(video_gs, video_mesh)]
        
        video_path = job_dir / "variant.mp4"
        imageio.mimsave(str(video_path), video, fps=30)
        
        # Save mesh
        mesh_path = job_dir / "variant_mesh.ply"
        mesh_result = outputs['mesh'][0]
        vertices_np = mesh_result.vertices.cpu().numpy()
        faces_np = mesh_result.faces.cpu().numpy()
        
        variant_mesh = o3d.geometry.TriangleMesh()
        variant_mesh.vertices = o3d.utility.Vector3dVector(vertices_np.astype(np.float64))
        variant_mesh.triangles = o3d.utility.Vector3iVector(faces_np.astype(np.int32))
        o3d.io.write_triangle_mesh(str(mesh_path), variant_mesh)
        
        jobs[job_id] = {
            "status": "completed",
            "type": "variant",
            "prompt": request.prompt,
            "cfg_strength": request.cfg_strength,
            "created_at": jobs[job_id]["created_at"],
            "completed_at": datetime.now().isoformat(),
            "video_url": f"/outputs/{job_id}/variant.mp4",
            "mesh_url": f"/outputs/{job_id}/variant_mesh.ply"
        }
        
        return jobs[job_id]
        
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/variant-batch")
async def generate_variant_batch(request: VariantBatchRequest):
    """Generate multiple variants with different cfg_strengths"""
    global current_base_mesh
    
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded")
    
    if current_base_mesh is None:
        raise HTTPException(status_code=400, detail="No base mesh loaded. Generate a 3D model first.")
    
    batch_id = str(uuid.uuid4())
    job_ids = []
    
    for cfg_strength in request.cfg_strengths:
        job_id = str(uuid.uuid4())
        jobs[job_id] = {
            "status": "processing",
            "type": "variant",
            "batch_id": batch_id,
            "prompt": request.prompt,
            "cfg_strength": cfg_strength,
            "created_at": datetime.now().isoformat()
        }
        job_ids.append(job_id)
    
    results = []
    
    for job_id, cfg_strength in zip(job_ids, request.cfg_strengths):
        try:
            print(f"Generating variant {cfg_strength} for batch {batch_id}")
            
            # Run variant generation
            outputs = pipeline.run_variant(
                current_base_mesh,
                request.prompt,
                seed=request.seed,
                slat_sampler_params={
                    "steps": request.steps,
                    "cfg_strength": cfg_strength,
                },
            )
            
            # Create job directory
            job_dir = outputs_dir / job_id
            job_dir.mkdir(exist_ok=True)
            
            # Render video
            video_gs = render_utils.render_video(outputs['gaussian'][0])['color']
            video_mesh = render_utils.render_video(outputs['mesh'][0])['normal']
            video = [np.concatenate([frame_gs, frame_mesh], axis=1) for frame_gs, frame_mesh in zip(video_gs, video_mesh)]
            
            video_path = job_dir / "variant.mp4"
            imageio.mimsave(str(video_path), video, fps=30)
            
            # Save mesh
            mesh_path = job_dir / "variant_mesh.ply"
            mesh_result = outputs['mesh'][0]
            vertices_np = mesh_result.vertices.cpu().numpy()
            faces_np = mesh_result.faces.cpu().numpy()
            
            variant_mesh = o3d.geometry.TriangleMesh()
            variant_mesh.vertices = o3d.utility.Vector3dVector(vertices_np.astype(np.float64))
            variant_mesh.triangles = o3d.utility.Vector3iVector(faces_np.astype(np.int32))
            o3d.io.write_triangle_mesh(str(mesh_path), variant_mesh)
            
            jobs[job_id] = {
                "status": "completed",
                "type": "variant",
                "batch_id": batch_id,
                "prompt": request.prompt,
                "cfg_strength": cfg_strength,
                "created_at": jobs[job_id]["created_at"],
                "completed_at": datetime.now().isoformat(),
                "video_url": f"/outputs/{job_id}/variant.mp4",
                "mesh_url": f"/outputs/{job_id}/variant_mesh.ply"
            }
            
            results.append(jobs[job_id])
            
        except Exception as e:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = str(e)
            results.append(jobs[job_id])
    
    return {
        "batch_id": batch_id,
        "total": len(job_ids),
        "results": results
    }


@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get the status of a job"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]


@app.get("/jobs")
async def list_jobs():
    """List all jobs"""
    return {"jobs": jobs}


@app.get("/outputs/{job_id}/{filename}")
async def get_output_file(job_id: str, filename: str):
    """Download an output file"""
    file_path = outputs_dir / job_id / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(file_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

