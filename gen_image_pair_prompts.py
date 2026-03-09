import numpy as np
from diffsynth_engine import fetch_model, QwenImagePipeline, QwenImagePipelineConfig
import math
import json
from typing import List, Tuple
from pydantic import TypeAdapter
from pydantic import ValidationError
from openai import OpenAI
from typing import Callable
import os
import csv
from dotenv import load_dotenv
load_dotenv()

def get_qwen_lora_pipeline():
    # loading models:
    model_path=fetch_model("Qwen/Qwen-Image-2512", path="transformer/*.safetensors")
    encoder_path=fetch_model("Qwen/Qwen-Image-2512", path="text_encoder/*.safetensors")
    vae_path=fetch_model("Qwen/Qwen-Image-2512", path="vae/*.safetensors")
    # print("model_path:", model_path)
    # print("encoder_path:", encoder_path)
    # print("vae_path:", vae_path)
    config = QwenImagePipelineConfig.basic_config(
        model_path=model_path,
        encoder_path=encoder_path,
        vae_path=vae_path,
        offload_mode="cpu_offload",
        device="cuda",
    )
    pipe = QwenImagePipeline.from_pretrained(config)
    # Load our turbo LoRA
    pipe.load_lora(
        path=fetch_model("Wuli-Art/Qwen-Image-2512-Turbo-LoRA", path="Wuli-Qwen-Image-2512-Turbo-LoRA-4steps-V1.0-bf16.safetensors"),
        scale=1.0,
        fused=True,
    )

    # Change scheduler config
    scheduler_config = {
        "exponential_shift_mu": math.log(2.5),
        "use_dynamic_shifting": True,
        "shift_terminal": None
    }
    pipe.apply_scheduler_config(scheduler_config)

    return pipe

def generate_image_qwen_lora(qwen_lora_pipeline: QwenImagePipeline, prompt: str):
    image = qwen_lora_pipeline(
        prompt=prompt,
        cfg_scale=1,
        num_inference_steps=8,  # 8 is also recommended
        seed=42,
        width=1328,
        height=1328
    )
    return image

def _parse_direction_groups(output_text: str, expected_count: int) -> List[List[str]]:
    try:
        payload = json.loads(output_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Expected JSON output, got: {output_text}") from exc

    if not isinstance(payload, dict) or "groups" not in payload:
        raise ValueError(f"Expected JSON object with 'groups', got: {payload}")

    try:
        adapter = TypeAdapter(List[Tuple[str, str, str]])
        groups = adapter.validate_python(payload["groups"])
    except ValidationError as exc:
        raise ValueError(f"Invalid groups format: {payload}") from exc

    if len(groups) != expected_count:
        raise ValueError(f"Expected {expected_count} groups, got {len(groups)}")

    return [list(item) for item in groups]


def direction_generator(basic_object: str, basic_positive_prompt: str, num_groups: int, for_text: bool = False):
    # use openai api to generate direction
    client = OpenAI() # defaults to os.environ.get("OPENAI_API_KEY")

    # v5 prompt
    sys_prompt = "You are a helpful assistant to generate prompts for image editing. Given an object and a basic positive attribute (e.g., 'a wooden chair, very modern style'), " + \
        "your goal is to generate a list of prompt groups. To generate each group, you need to follow the instruction below:\n" + \
        "1. identify the basic object O (e.g., 'a chair') by removing all discriptives (e.g., color, material, style, etc.) from the given object. \n" + \
        "2. based on the basic positive attribute, generate a simple but concrete, description of the positive attribute to the maximum extent specific to the basic object O using less than 10 words (P). \n" + \
        "3. based on the basic positive attribute, generate a simple but concrete, description of the completely opposite attribute of that positive attribute to the maximum extent specific to the basic object O using less than 10 words (N). \n" + \
        "4. generate two attributes A1 and A2 in different dimensions that can describe the basic object, but perpendicular and irrelevant to the positive attribute P and the negative attribute N. \n\n" + \
        f"Following the above steps, generate {num_groups} groups of prompts, each group containing a neutral prompt T0, a positive prompt P and a negative prompt N. T0 consists of the basic object O, A1, and A2. \n" + \
        ("the elements in the neutral prompt T0 should be organized such that the resulting image will be a picture with only the basic object (with corresponding attributes) in the middle with a black background. \n\n" if not for_text else "\n") + \
        "For example, if the basic object is 'a blue sofa made of leather' and the positive attribute is 'modern style', a possible output group can be: \n" + \
        ("['a picture with only a sofa in the middle with a black background, the sofa is yellow, tall and narrow', 'the sofa is very modern and minimalist style', 'the sofa is very traditional and ornate style']\n\n" if not for_text else \
        "['a yellow, tall and narrow sofa', 'the sofa is very modern and minimalist style', 'the sofa is very traditional and ornate style']\n\n")  + \
        "Explanation: \n" + \
        "1. 'a sofa' is O, the basic object by removing the discriptives from the given object. \n" + \
        "1. 'the sofa is very modern and minimalist style' is P, the positive attribute description to the maximum extent specific to the basic object O. \n" + \
        "2. 'the sofa is very traditional and ornate style' is N, the negative attribute description to the maximum extent specific to the basic object O. \n" + \
        "3. 'yellow color' and 'tall and narrow' are A1 and A2, the attributes that are perpendicular and irrelevant to the positive/negative attribute. \n\n" + \
        "Return JSON with the schema: {\"groups\": [[T0, P, N], ...]} where T0, P, and N are strings.\n"



    results = []
    user_prompt = (
        "Please generate the list of prompt groups for the following basic object and basic positive attribute: \n"
        f"Basic object: '{basic_object}' \n"
        f"Basic positive attribute: '{basic_positive_prompt}' \n"
    )
    print("sys_prompt:", sys_prompt)
    response = client.chat.completions.create(
        model="gpt-4.1",
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
                "name": "direction_groups",
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "groups": {
                            "type": "array",
                            "minItems": num_groups,
                            "maxItems": num_groups,
                            "items": {
                                "type": "array",
                                "minItems": 3,
                                "maxItems": 3,
                                "items": {"type": "string"},
                            },
                        }
                    },
                    "required": ["groups"],
                },
            },
        },
    )
    groups = _parse_direction_groups(response.choices[0].message.content, num_groups)
    results.extend(groups)
    return results

def get_image_pairs_from_object_and_pos_attributes(img_gen_pipeline, img_gen_method: Callable, obj_name: str, pos_attribute: str, num_pairs: int, out_dir: str, for_text: bool = False):
    # use sdxl to generate image for the object and the attributes
    subfolder = f"{img_gen_method.__name__}_{obj_name.replace(' ', '_')}_{pos_attribute.replace(' ', '_')}"
    os.makedirs(os.path.join(out_dir, subfolder), exist_ok=True)
    file_prompt_map = {}
    slider_prompts = direction_generator(obj_name, pos_attribute, num_pairs, for_text=for_text)
    for i, slider_prompt in enumerate(slider_prompts):
        img_gen_neutral_prompt, pos_prompt, neg_prompt = slider_prompt
        # print('Generating image pairs for:', slider_prompt)
        img_gen_pos_prompt = f"{img_gen_neutral_prompt}, {pos_prompt}, 4k"
        img_gen_neg_prompt = f"{img_gen_neutral_prompt}," + (f" {neg_prompt}, 4k" if neg_prompt else " 4k")
        print('Generating positive image for:', img_gen_pos_prompt)
        img_gen_pos_img = img_gen_method(img_gen_pipeline, img_gen_pos_prompt)
        img_gen_pos_img.save(os.path.join(out_dir, subfolder, f"{i}_pos.png"))
        print('Generating negative image for:', img_gen_neg_prompt)
        img_gen_neg_img = img_gen_method(img_gen_pipeline, img_gen_neg_prompt)
        img_gen_neg_img.save(os.path.join(out_dir, subfolder, f"{i}_neg.png"))
        file_prompt_map[f"{i}_pos.png"] = img_gen_pos_prompt
        file_prompt_map[f"{i}_neg.png"] = img_gen_neg_prompt
    
    # output the file_prompt_map to a tsv file
    with open(os.path.join(out_dir, subfolder, "img_gen_prompt_map.tsv"), "w", encoding='utf-8') as f:
        for file, prompt in file_prompt_map.items():
            f.write(f"{file}\t{prompt}\n")


def batch_get_image_pairs_from_object_and_pos_attributes(img_gen_pipeline: Callable, img_gen_method: Callable, obj_name_pos_attribute_pairs: List[Tuple[str, str]], num_pairs: int, out_dir: str, for_text: bool = False):
    for obj_name, pos_attribute in obj_name_pos_attribute_pairs:
        get_image_pairs_from_object_and_pos_attributes(img_gen_pipeline, img_gen_method, obj_name, pos_attribute, num_pairs, out_dir, for_text=for_text)


if __name__ == "__main__":
    obj_name_pos_attribute_pairs = []
    dataset_name = "../3d-interior-val/new_dataset"
    with open("../3d-interior-val/new_validation.csv", "r", encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader) # skip the header
        for row in reader:
            basic_description, attributes, asset_name = row
            print('Adding object:', basic_description, 'with attributes:', attributes)
            pos_attribute = attributes
            obj_name_pos_attribute_pairs.append((basic_description, pos_attribute))
    out_dir = "outputs/test_img_slat/img_direction_pairs_new_dataset"
    qwen_lora_pipeline = get_qwen_lora_pipeline()
    batch_get_image_pairs_from_object_and_pos_attributes(qwen_lora_pipeline, generate_image_qwen_lora, obj_name_pos_attribute_pairs[3:], 12, out_dir)