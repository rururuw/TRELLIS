
import numpy as np
from transformers import CLIPProcessor, CLIPModel
import torch
import os
from typing import List, Tuple
from PIL import Image
import csv
import shutil
import base64
import io
import json
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
import ImageReward as RM
import open_clip
import time
from google import genai
from google.genai import types

def get_CLIP_model_and_processor(device="cuda"):
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", use_safetensors=True)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14", use_fast=True)
    model = model.to(device).eval()
    return model, processor

def get_open_clip_model_and_processor(device="cuda"):
    model, preprocess = open_clip.create_model_from_pretrained('hf-hub:laion/CLIP-ViT-g-14-laion2B-s12B-b42K')
    tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-g-14-laion2B-s12B-b42K')
    return model, tokenizer, preprocess

def get_ImageReward_model():
    model = RM.load("ImageReward-v1.0")
    return model

def calc_CLIP_similarity_text_vs_images(model, processor, text_prompt, images: List[Image.Image], device="cuda"):
    # Batch process all images at once
    inputs = processor(text=[text_prompt], images=images, return_tensors="pt", padding=True)
    # Move inputs to same device as model
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    # logits_per_image shape: [num_images, 1] - similarity of each image to the single text
    clip_scores = outputs.logits_per_image[:, 0].detach().cpu().numpy()
    return np.mean(clip_scores)


def relevant_view_prompt_generator(basic_editing_prompt: str):
    # use openai api to generate direction
    client = OpenAI() # defaults to os.environ.get("OPENAI_API_KEY")
    sys_prompt = (
        "I have hundreds of images taken from different angles around an 3D object (like a 360-degree orbit). "
        "I need to find the best images to perform a specific edit. The problem is that my current search tool gets confused. "
        "For example, if I ask for views with 'front lights,' it might accidentally show me views with 'rear lights' because they both look like 'lights'."
        "You are an expert in Computer Vision and 3D Asset Editing."
        "Your task is to generate contrastive text descriptions to help a CLIP model filter rendered viewpoints of a 3D object, based on the given editing prompt.\n\n"
        "Rules for your response: "
        "1. Be extremely concise: Use 15 words or fewer per description."
        "2. Focus on content: Describe exactly what parts should be visible in the frame."
        "3. No camera jargon: Do not use words like 'camera,' 'viewpoint,' 'angle,' or 'zoom.'"
        "4. If the whole object is to be changed, the target description should be a description of the most visually representative parts of the object, "
        "and the negative description should be something along the lines of 'the background' or 'no visible object'."
        "Important: The description must focus on the specific PART of the object that should be changed. "
        "Do NOT describe the final edited result or attributes of the edited object. "
        "Do NOT add unnecessary details about the object's meterial, color, texture, etc, so that the description can be generalized to other objects of the same type.\n"
        "Please provide: "
        "1. Target description: Describe the specific part of the object being edited so it can be identified. "
        "2. Negative description: Describe the part of the object that is farthest away from the edit or most likely to be confused with it.\n\n"
        "Example: \n"
        "For editing prompt: 'Make the dog's tail very long.' \n"
        "Target description can be: 'The dog's bushy tail and back legs.' "
        "Negative description can be: 'The dog's nose, eyes, and front paws.' \n"
        "For editing prompt: 'Make the car's front light very big and round.' \n"
        "Target description can be: 'The front headlights and car grille.' "
        "Negative description can be: 'The rear taillights and exhaust pipes.' \n\n"
        "For editing prompt: 'Make the sofa very modern and stylish.' \n"
        "Target description can be: 'Sofa arms, front panel, legs, and backrest.' "
        "Negative description can be: 'Empty background, no sofa visible.' \n\n"
        "Return JSON with the schema: {\"prompts\": [negative_prompt, target_prompt]} where negative_prompt and target_prompt are strings."
    )
    user_prompt = (
        f"Editing Prompt: {basic_editing_prompt} \n"
    )
    response = client.chat.completions.create(
        model="gpt-5.2",
        messages=[
            {
                "role": "system",
                "content": sys_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=0.3,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "relevant_view_prompts",
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "prompts": {
                            "type": "array",
                            "minItems": 2,
                            "maxItems": 2,
                            "items": {"type": "string"},
                        }
                    },
                    "required": ["prompts"],
                },
            },
        },
    )
    payload = json.loads(response.choices[0].message.content)
    prompts = payload["prompts"]
    assert len(prompts) == 2, f"Expected 2 prompts, got {len(prompts)}"
    return prompts  # [negative_prompt, positive_prompt]

def image_content_checker(positive_content: str, image_path: str):
    # if positive_content is in the image, and negative_content is not in the image, return True
    # otherwise return False
    # use openai api to check the image content
    client = OpenAI()

    # Encode the image as a base64 data URL
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")
    image_url = f"data:image/png;base64,{img_b64}"

    response = client.chat.completions.create(
        model="gpt-5.2",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an image content checker. You are given a description of the content and an image. "
                    "You need to check if the description is present in the image. "
                    "If the description is present in the image, return {\"result\": true}, otherwise return {\"result\": false}."
                )
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"content description: {positive_content}\n"
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    },
                ],
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "image_content_checker",
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "result": {"type": "boolean"}
                    },
                    "required": ["result"],
                },
            },
        },
    )
    payload = json.loads(response.choices[0].message.content)
    return payload["result"]

def editing_prompt_view_filtering(client: OpenAI, editing_prompt: str, image_path: str):
    # Encode the image as a base64 data URL
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")
    image_url = f"data:image/png;base64,{img_b64}"

    response = client.chat.completions.create(
        model="gpt-5.2",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a view-relevance checker for 3D object editing. "
                    "You will be given an editing prompt describing a desired change to a 3D object, "
                    "and an image showing the object rendered from a particular viewpoint.\n\n"
                    "Your task is to determine whether the region or part of the object targeted by the editing prompt "
                    "is clearly visible in this view.\n\n"
                    "Guidelines:\n"
                    "- If the edit targets a specific part (e.g., \"make the eyes bigger\"), check whether that part "
                    "is visible and not occluded or facing away from the camera.\n"
                    "- If the edit targets the entire object (e.g., \"change the color/style of the object\"), check whether the object "
                    "is sufficiently visible in the image.\n\n"
                    "Return {\"result\": true} if the targeted region is clearly visible, "
                    "or {\"result\": false} otherwise."
                )
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Editing prompt: {editing_prompt}\n"
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    },
                ],
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "editing_prompt_view_filtering",
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "result": {"type": "boolean"}
                    },
                    "required": ["result"],
                },
            },
        },
    )
    payload = json.loads(response.choices[0].message.content)
    return payload["result"]

def direction_generator_edit(basic_object: str, basic_attribute: str,):
    # use openai api to generate direction
    client = OpenAI() # defaults to os.environ.get("OPENAI_API_KEY")

    # sys prompt v2.1
    sys_prompt = (
        "You are an expert assistant specializing in generating concise, high-impact prompts for image editing APIs. "
        "Given an 'Object A' and an 'Attribute B', your task is to generate a 'negative_prompt' and a 'positive_prompt' "
        "that modify the object along the spectrum of that attribute.\n\n"
        
        "### INSTRUCTIONS:\n"
        "1. Positive Prompt: Edit Object A to present the MAXIMUM extreme or highest intensity of Attribute B.\n"
        "2. Negative Prompt: Edit Object A to present the MINIMUM extreme or exact opposite of Attribute B.\n"
        "3. Sentiment Independence: Note that Attribute B can itself be inherently negative (e.g., 'rusty'), neutral, or positive. "
        "The 'positive prompt' always means MORE of the attribute, and the 'negative prompt' always means LESS of it.\n"
        "4. Brevity (CRITICAL): Image models lose focus with long prompts. Keep the editing description concise and direct (under 20 words).\n\n"
        # "5. Context & Reassurance: ALWAYS end the generated prompt with this exact, concise phrasing: 'Keep all other elements unchanged. Skip edit if not visible.'\n\n"
        
        "### SAFETY GUARDRAILS (CRITICAL):\n"
        "- NEVER generate prompts that could trigger NSFW, nudity, or sexual content safety filters.\n"
        "- When reducing clothing, use safe, concise descriptors like 'casual summer wear' or 'lightweight clothing'. Avoid 'skimpy', 'revealing', 'bare', or 'underwear'.\n"
        "- Modifying physical traits ('very thin', 'highly muscular') is perfectly safe, but keep it purely descriptive and non-suggestive.\n\n"
        
        "### EXAMPLES:\n"
        "Object: 'a blue sofa' | Attribute: 'modern style'\n"
        "Negative: 'Make the sofa traditional, antique, and ornate style.'\n"
        "Positive: 'Make the sofa ultra-modern, sleek, and minimalist style.'\n\n"
        
        "Object: 'a woman' | Attribute: 'eye size'\n"
        "Negative: 'Make the woman's eyes extremely small and narrow.'\n"
        "Positive: 'Make the woman's eyes dramatically large and wide open.'\n\n"

        "Object: 'a woman' | Attribute: 'wearing very warm and thick clothes'\n"
        "Negative: 'Make the woman wear modest, casual summer clothes like a t-shirt.'\n"
        "Positive: 'Make the woman wear extremely warm, heavy winter coats and thick scarves.'\n\n"
        
        "### OUTPUT FORMAT:\n"
        "Return ONLY a valid JSON object matching this schema without any markdown formatting:\n"
        "{\"prompts\": [\"<negative_prompt>\", \"<positive_prompt>\"]}"
    )
    user_prompt = (
        f"Object: {basic_object} \n"
        f"Attribute: {basic_attribute} \n"
    )
    response = client.chat.completions.create(
        model="gpt-5.2",
        messages=[
            {
                "role": "system",
                "content": sys_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=0.3,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "direction_edit_prompts",
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "prompts": {
                            "type": "array",
                            "minItems": 2,
                            "maxItems": 2,
                            "items": {"type": "string"},
                        }
                    },
                    "required": ["prompts"],
                },
            },
        },
    )
    payload = json.loads(response.choices[0].message.content)
    prompts = payload["prompts"]
    assert len(prompts) == 2, f"Expected 2 prompts, got {len(prompts)}"
    return prompts  # [negative_prompt, positive_prompt]

def generating_prompt_pairs_from_object_and_attribute_new_validation(validation_file: str, out_dir: str, start_index: int = None):
    task_seq = []
    with open(validation_file, "r", encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader) # skip the header
        for row in reader:
            basic_description, attributes, asset_name = row            
            print('adding task: object:', basic_description, 'with attributes:', attributes, 'asset name:', asset_name)
            task_seq.append((basic_description, attributes, asset_name))
    os.makedirs(out_dir, exist_ok=True)
    prompt_pairs = []
    for i, (basic_description, attributes, asset_name) in enumerate(task_seq):
        if start_index is not None and i < start_index:
            print(f"Skipping task {i} because it is less than start_index {start_index}")
            continue
        print('processing task: object:', basic_description, 'with attributes:', attributes, 'asset name:', asset_name)
        neg_prompt, pos_prompt = direction_generator_edit(basic_description, attributes)
        print('negative prompt:', neg_prompt, 'positive prompt:', pos_prompt)
        prompt_pairs.append((neg_prompt, pos_prompt))
    with open(os.path.join(out_dir, "new_val_prompt_pairs_v3.tsv"), "w", encoding='utf-8') as f:
        for neg_prompt, pos_prompt in prompt_pairs:
            f.write(f"{neg_prompt}\t{pos_prompt}\n")
    print('Done generating image pairs for all tasks')

def get_relevant_view_prompts(asset_info_path: str, prompt_pairs_path: str, out_path: str):
    asset_info = []
    prompt_pairs = []
    with open(prompt_pairs_path, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            neg_prompt, pos_prompt = row
            prompt_pairs.append((neg_prompt.strip(), pos_prompt.strip()))
    with open(asset_info_path, 'r') as f:
        reader = csv.reader(f)
        # skip the first row
        next(reader)
        for row in reader:
            asset_name, basic_description, attributes = row
            asset_info.append((asset_name, basic_description, attributes))
    resulting_prompts = []
    for i, (asset_name, basic_description, attributes) in enumerate(asset_info):
        print(f"Processing asset {asset_name} with editing prompt {basic_description}")
        neg_prompt, pos_prompt = prompt_pairs[i]
        relevant_view_prompts = relevant_view_prompt_generator(pos_prompt)
        print(f"Relevant view prompts for {asset_name}: {relevant_view_prompts}")
        resulting_prompts.append(relevant_view_prompts)
    with open(out_path, 'w') as f:
        for prompt in resulting_prompts:
            f.write(f"{prompt[0]}\t{prompt[1]}\n")

def select_best_views(model, processor, view_image_dir: str, edit_prompt: str, sort_by_score=True, device="cuda"):
    view_image_paths = [os.path.join(view_image_dir, f) for f in os.listdir(view_image_dir) if f.endswith(".png")]
    view_image_paths.sort()
    view_images = [Image.open(f) for f in view_image_paths]
    inputs = processor(text=[edit_prompt], images=view_images, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    # logits_per_image shape: [num_images, 1] - similarity of each image to the single text
    clip_scores = outputs.logits_per_image[:, 0].detach().cpu().numpy()
    clip_scores_items = [(view_image_paths[i], clip_scores[i]) for i in range(len(view_image_paths))]
    if sort_by_score:
        # sort the clip scores dict by the clip score in descending order
        clip_scores_items = sorted(clip_scores_items, key=lambda x: x[1], reverse=True)
    return clip_scores_items

def select_best_views_by_ImageReward(model, prompt: str, view_image_dir: str):
    view_image_paths = [os.path.join(view_image_dir, f) for f in os.listdir(view_image_dir) if f.endswith(".png")]
    view_image_paths.sort()
    scores = model.score(prompt, view_image_paths)
    scores_items = [(view_image_paths[i], scores[i]) for i in range(len(view_image_paths))]
    return scores_items

def get_N_verified(client: OpenAI, editing_prompt: str, final_scores: List[Tuple[str, float]], N: int):
    # the goal is to get N verified good images from the ranked list of views
    verified_good_images = []
    for i, (view_image_path, score) in enumerate(final_scores):
        print(f"Checking image {view_image_path} with editing prompt: {editing_prompt}...")
        time_start = time.time()
        result = editing_prompt_view_filtering(client, editing_prompt, view_image_path)
        total_time = time.time() - time_start
        print(f"Result: {result}")
        print(f"Time taken: {total_time:.2f} seconds")
        if result:
            verified_good_images.append((view_image_path, score))
            if len(verified_good_images) >= N:
                break
    print(f"Found {len(verified_good_images)} verified good images after checking {i+1} views")
    return verified_good_images

def get_best_views_by_open_clip(model, preprocessor, tokenizer, text_prompt, view_image_dir: str, device="cuda"):
    # Batch process all images at once
    view_image_paths = [os.path.join(view_image_dir, f) for f in os.listdir(view_image_dir) if f.endswith(".png")]
    view_image_paths.sort()
    images = [preprocessor(Image.open(view_image_path)) for view_image_path in view_image_paths]
    images = torch.stack(images) # shape: [num_images, 3, 224, 224]
    text = tokenizer([text_prompt])
    with torch.no_grad(), torch.amp.autocast(device):
        image_features = model.encode_image(images)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * text_features @ image_features.T).softmax(dim=-1).detach().cpu().numpy()
    # return the mean of the similarity scores
    scores_items = [(view_image_paths[i], similarity[0][i]) for i in range(len(view_image_paths))]
    return scores_items

def get_best_views_from_affected_view_prompts(affected_view_prompts_path: str, asset_info_path: str, view_image_dir: str, out_dir: str, device="cuda"):
    model, processor = get_CLIP_model_and_processor(device)
    best_views = []
    affected_view_prompts = []
    asset_info = []
    os.makedirs(out_dir, exist_ok=True)
    with open(affected_view_prompts_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            affected_view_prompt = line.strip()
            affected_view_prompts.append(affected_view_prompt)
    with open(asset_info_path, 'r') as f:
        reader = csv.reader(f)
        # skip the first row
        next(reader)
        for row in reader:
            basic_description,attributes,asset_name = row
            asset_info.append((asset_name, basic_description, attributes))

    for i, (asset_name, basic_description, attributes) in enumerate(asset_info):
        print(f"Processing asset {asset_name} with affected view prompt {affected_view_prompts[i]}")
        affected_view_prompt = affected_view_prompts[i]
        view_image_dir_asset = os.path.join(view_image_dir, asset_name[:-4] + '_views')
        best_views = select_best_views(model, processor, view_image_dir_asset, affected_view_prompt)
        best_views_10 = best_views[:10]
        sub_out_dir = os.path.join(out_dir, asset_name[:-4] + '_best_views_avp')
        os.makedirs(sub_out_dir, exist_ok=True)
        for j, (view_path, score) in enumerate(best_views_10):
            shutil.copy(view_path, os.path.join(sub_out_dir, f'best_{j}_{os.path.basename(view_path).split(".")[0]}_{score:.2f}.png'))

def get_best_views_from_relevant_view_prompts(
    prompt_pairs_path: str, 
    asset_info_path: str, 
    view_image_dir: str, 
    out_dir: str,  
    method: str = 'RM',
    verify: bool = False,
    N: int = 10
):
    # model, processor = get_CLIP_model_and_processor(device)
    if method == 'RM':
        model = get_ImageReward_model()
    elif method == 'CLIP':
        model, processor = get_CLIP_model_and_processor()
    elif method == 'OpenCLIP':
        model, tokenizer, preprocessor = get_open_clip_model_and_processor()
    else:
        raise ValueError(f"Invalid method: {method}")
    client_openai = OpenAI()
    prompt_pairs = []
    asset_info = []
    os.makedirs(out_dir, exist_ok=True)
    with open(prompt_pairs_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            negative_prompt, target_prompt = line.strip().split('\t')
            prompt_pairs.append((negative_prompt.strip(), target_prompt.strip()))
    with open(asset_info_path, 'r') as f:
        reader = csv.reader(f)
        # skip the first row
        next(reader)
        for row in reader:
            basic_description,attributes,asset_name = row
            asset_info.append((asset_name, basic_description, attributes))

    for i, (asset_name, basic_description, attributes) in enumerate(asset_info):
        print(f"Processing asset {asset_name} with prompt pairs {prompt_pairs[i]}")
        negative_prompt, target_prompt = prompt_pairs[i]
        view_image_dir_asset = os.path.join(view_image_dir, asset_name[:-4] + '_views')
        sub_out_dir = os.path.join(out_dir, asset_name[:-4] + f'_best_views_{i}')
        if os.path.exists(sub_out_dir) and len(os.listdir(sub_out_dir)) > 0:
            print(f"{sub_out_dir} already exists and is not empty. Skipping {i}th task")
            continue
        os.makedirs(sub_out_dir, exist_ok=True)
        if method == 'RM':
            best_views_target = select_best_views_by_ImageReward(model, target_prompt, view_image_dir_asset)
        elif method == 'CLIP':
            best_views_target = select_best_views(model, processor, view_image_dir_asset, target_prompt, sort_by_score=False)
        elif method == 'OpenCLIP':
            best_views_target = get_best_views_by_open_clip(model, preprocessor, tokenizer, target_prompt, view_image_dir_asset)
        else:
            raise ValueError(f"Invalid method: {method}")
        final_scores = best_views_target
        final_scores.sort(key=lambda x: x[1], reverse=True)
        if verify:
            verified_good_images = get_N_verified(client_openai, target_prompt, final_scores, N)
        else:
            verified_good_images = final_scores[:N]
        for j, (view_path, score) in enumerate(verified_good_images):
            shutil.copy(view_path, os.path.join(sub_out_dir, f'best_{j}_{os.path.basename(view_path).split(".")[0]}_{score:.2f}.png'))

def edit_image_gpt_image(client: OpenAI, input_image_path: str, edit_prompt: str):
    # Edit the image using GPT-Image
    result = client.images.edit(
        model="gpt-image-1.5",
        image=[
            open(input_image_path, "rb"),
        ],
        prompt=edit_prompt
    )
    image_base64 = result.data[0].b64_json
    image_bytes = base64.b64decode(image_base64)
    # convert image_bytes to Image
    image = Image.open(io.BytesIO(image_bytes))
    return image

def edit_image_gemini(client: genai.Client, model_name: str, input_image_path: str, edit_prompt: str):
    # Edit the image using Gemini
    image = Image.open(input_image_path)
    response = client.models.generate_content(
        model=model_name,
        contents=[edit_prompt, image],
    )
    edited_image = None
    for part in response.parts:
        if part.text is not None:
            print(part.text)
        elif part.inline_data is not None:
            image = part.as_image()
            edited_image = image
    return edited_image

def editing_best_views(
    best_views_path: str, 
    prompt_pairs_path: str, 
    asset_info_path: str, 
    out_dir: str, 
    method: str = 'gemini-2.5-flash-image'
):
    if method == 'GPT':
        client = OpenAI()
    elif 'gemini' in method:
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    else:
        raise ValueError(f"Invalid method: {method}")
    prompt_pairs = []
    asset_info = []
    os.makedirs(out_dir, exist_ok=True)
    with open(prompt_pairs_path, "r", encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            neg_prompt, pos_prompt = row
            prompt_pairs.append((neg_prompt, pos_prompt))
    with open(asset_info_path, 'r') as f:
        reader = csv.reader(f)
        # skip the first row
        next(reader)
        for row in reader:
            basic_description,attributes,asset_name = row
            asset_info.append((asset_name, basic_description, attributes))
    reassurance_prompt = (
        " Do not change any other attributes of the object and its background that are irrelevant to the editing. "
        "If the part that is supposed to be edited is not visible in the image, do not edit it."
    )
    for i, (neg_prompt, pos_prompt) in enumerate(prompt_pairs):
        asset_name, basic_description, attributes = asset_info[i]
        best_view_folder_path = os.path.join(best_views_path, asset_name[:-4] + f'_best_views_{i}')
        best_view_paths = [os.path.join(best_view_folder_path, f) for f in os.listdir(best_view_folder_path) if f.endswith(".png")]
        best_view_paths.sort(key=lambda x: int(os.path.basename(x).split("_")[1]))
        sub_out_dir = os.path.join(out_dir, asset_name[:-4] + f'_edited_views_{i}')
        if os.path.exists(sub_out_dir) and len(os.listdir(sub_out_dir)) > 0:
            print(f"{sub_out_dir} already exists and is not empty. But not Skipping.")
        else:
            os.makedirs(sub_out_dir, exist_ok=True)
        for best_view_path in best_view_paths:
            # check if edited image already exists
            edited_name = '.'.join(os.path.basename(best_view_path).split(".")[:-1]) + '_edited_neg.png'
            if os.path.exists(os.path.join(sub_out_dir, edited_name)):
                print(f"Skipping {os.path.join(sub_out_dir, edited_name)} because it already exists")
                continue
            print(f"Editing view {os.path.basename(best_view_path)} with negative prompt {neg_prompt}")
            start_time = time.time()
            if method == 'GPT':
                edited_image = edit_image_gpt_image(client, best_view_path, neg_prompt + reassurance_prompt)
            elif 'gemini' in method:
                edited_image = edit_image_gemini(client, method, best_view_path, neg_prompt + reassurance_prompt)
            else:
                raise ValueError(f"Invalid method: {method}")
            edited_image.save(os.path.join(sub_out_dir, edited_name)) 
            end_time = time.time()
            print(f"Time: {end_time - start_time} seconds")
            print(f"Editing view {os.path.basename(best_view_path)} with positive prompt {pos_prompt}")
            edited_name = '.'.join(os.path.basename(best_view_path).split(".")[:-1]) + '_edited_pos.png'
            start_time = time.time()
            if method == 'GPT':
                edited_image = edit_image_gpt_image(client, best_view_path, pos_prompt + reassurance_prompt)
            elif 'gemini' in method:
                edited_image = edit_image_gemini(client, method, best_view_path, pos_prompt + reassurance_prompt)
            else:
                raise ValueError(f"Invalid method: {method}")
            edited_image.save(os.path.join(sub_out_dir, edited_name))
            end_time = time.time()
            print(f"Time: {end_time - start_time} seconds")


if __name__ == "__main__":
    affected_view_prompts_path = "new_val_affected_view_prompts.txt"
    prompt_pairs_path = "new_val_prompt_pairs_v2.tsv"
    relevant_view_prompts_path = "new_val_relevant_view_prompts.tsv"
    asset_info_path = "new_validation.csv"
    num_views = 40
    view_image_dir = f"multi_views_new_dataset_{num_views}"

    # for prepareing for a slider:
    # 0. render camera views from the object (in render_multiviews_blender_mp.py)
    # 1. generate prompt pairs
    # generating_prompt_pairs_from_object_and_attribute_new_validation(
    #     validation_file=asset_info_path,
    #     out_dir='./',
    #     start_index=40
    # )
    # 2. get best views
    # get_best_views_from_relevant_view_prompts(
    #     prompt_pairs_path, 
    #     asset_info_path, 
    #     view_image_dir, 
    #     f'new_val_best_views_RM_edppv2_mv{num_views}_verify_gpt52', 
    #     method='RM', 
    #     verify=True
    # )
    # 3. edit best views
    editing_best_views(
        best_views_path=f'new_val_best_views_RM_edppv2_mv{num_views}_verify_gpt52', 
        prompt_pairs_path=prompt_pairs_path, 
        asset_info_path=asset_info_path, 
        out_dir=f'new_val_edited_views_RM_edppv2_mv{num_views}_verify_gpt52', 
        # method='gemini-2.5-flash-image'
        method='GPT'
    )