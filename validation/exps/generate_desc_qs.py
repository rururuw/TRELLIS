import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from slider_mapping import get_questions_VQA, get_views_for_eval, describe_objs_from_views, filter_questions
import csv
import json


def read_all_assets(asset_info_path: str, assets_dir: str):
    assets = []
    with open(asset_info_path, 'r') as f:
        reader = csv.reader(f)
        # skip the header
        next(reader)
        for row in reader:
            obj, attr, asset_id = row
            assets.append((obj, attr, os.path.join(assets_dir, asset_id)))
    return assets

def read_edit_prompts_pairs(edit_prompts_pairs_path: str):
    edit_prompts_pairs = []
    with open(edit_prompts_pairs_path, 'r') as f:
        # it's a tsv file
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            neg_prompt, pos_prompt = row
            edit_prompts_pairs.append((neg_prompt, pos_prompt))
    return edit_prompts_pairs

def generate_desc_qs(assets: list, edit_prompts_pairs: list, out_dir: str):
    print(f"Generating descriptions and questions for {len(assets)} assets")
    print(f"Editing prompts pairs: {len(edit_prompts_pairs)}")
    for i, asset in enumerate(assets):
        obj, attr, asset_path = asset
        neg_prompt, pos_prompt = edit_prompts_pairs[i]
        print(f"Processing asset {i+1} of {len(assets)}: {obj} {attr}")
        # create a new directory for the asset
        asset_dir = os.path.join(out_dir, os.path.basename(asset_path[:-4]))
        os.makedirs(asset_dir, exist_ok=True)
        # get views for evaluation
        views_dir = os.path.join(asset_dir, "views")
        if os.path.exists(views_dir) and len(os.listdir(views_dir)) >= 4:
            print(f"Views already exist for asset {i+1} of {len(assets)}")
            if f"info_{i}.json" in os.listdir(asset_dir):
                print(f"Info already exists for asset {i+1} of {len(assets)}. Skipping (3/6/2026)...")
                continue
        else:
            get_views_for_eval(asset_path, views_dir)
        # generate description of the object
        view_paths = [os.path.join(views_dir, f) for f in os.listdir(views_dir) if f.endswith(".png")]
        # special case for reused assets:
        if any(f.endswith(".json") for f in os.listdir(asset_dir)):
            print(f"Info already exists for asset {i+1} of {len(assets)}: Reuse description from other obj-attr pairs")
            info_file = sorted([f for f in os.listdir(asset_dir) if f.endswith(".json")])[0]
            with open(os.path.join(asset_dir, info_file), 'r') as f:
                info = json.load(f)
                description = info["description"]
                qid2dependencies = info["dependencies"]
                qid2questions = info["questions"]
        else:
            # generate description of the object
            description = describe_objs_from_views(view_paths)
            # generate questions based on the description
            qid2dependencies, qid2questions = get_questions_VQA(description)
            # get number of questions and dependencies
            num_questions = len(qid2questions)
            num_dependencies = len(qid2dependencies)
            if num_questions != num_dependencies:
                print(f"!!!! Number of questions and dependencies do not match for asset {i+1} of {len(assets)}")
        # filter questions based on the descriptions
        print(f"Filtering questions for asset {i+1} of {len(assets)}")
        removed_questions, filtered_questions = filter_questions(list(qid2questions.values()), pos_prompt)
        # save the questions
        with open(os.path.join(asset_dir, f"info_{i}.json"), 'w') as f: # new change after realizing there are multiple obj-attr pairs for the same asset
            json.dump({
                "description": description,
                "editing_prompt": pos_prompt,
                "questions": qid2questions,
                "dependencies": qid2dependencies,
                "removed_questions": removed_questions,
                "filtered_questions": filtered_questions,
            }, f, indent=4)

if __name__ == "__main__":
    assets_dir = '/home/rwang/TRELLIS/validation/new_dataset'
    assets_info_path = '/home/rwang/TRELLIS/validation/new_validation.csv'
    edit_prompt_pairs_path = '/home/rwang/TRELLIS/validation/new_val_prompt_pairs_v2.tsv'
    out_dir = "/data/ru_data/results/trellis_output/validation/test_slider_preparation"
    assets = read_all_assets(assets_info_path, assets_dir)
    edit_prompts_pairs = read_edit_prompts_pairs(edit_prompt_pairs_path)
    generate_desc_qs(assets, edit_prompts_pairs, out_dir)