import os
import shutil

SRC = os.path.join(os.path.dirname(__file__),
                   "new_val_edited_views_RM_edppv2_mv40_verify_gpt52")
DST = '/data/ru_data/sliders/dataset_images'

for subfolder in sorted(os.listdir(SRC)):
    new_subfolder = subfolder.split('_')[-1]
    src_sub = os.path.join(SRC, subfolder)
    if not os.path.isdir(src_sub):
        continue

    pos_dir = os.path.join(DST, new_subfolder, "pos")
    neg_dir = os.path.join(DST, new_subfolder, "neg")
    os.makedirs(pos_dir, exist_ok=True)
    os.makedirs(neg_dir, exist_ok=True)

    for fname in sorted(os.listdir(src_sub)):
        src_path = os.path.join(src_sub, fname)
        if not os.path.isfile(src_path):
            continue
        if "_neg" in fname:
            new_fname = fname.replace("_neg", "")
            shutil.copy2(src_path, os.path.join(neg_dir, new_fname))
        elif "_pos" in fname:
            new_fname = fname.replace("_pos", "")
            shutil.copy2(src_path, os.path.join(pos_dir, new_fname))

    print(f"{new_subfolder}: {len(os.listdir(pos_dir))} pos, {len(os.listdir(neg_dir))} neg")

print(f"\nDone. Output: {DST}")
