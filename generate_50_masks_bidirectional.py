import os
import sys
import json
import re
import torch
import numpy as np
import SimpleITK as sitk
from glob import glob
from os.path import join, basename
from skimage import measure
from PIL import Image

# Define Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT = join(BASE_DIR, 'MedSAM2/checkpoints/MedSAM2_latest.pt')
IMGS_PATH = "/data1/students/manolina/MedSAM2/50_image/"
MODEL_CFG = "configs/sam2.1_hiera_t512.yaml"
PRED_SAVE_DIR = "/data1/students/manolina/50_generated_mask_bidirectional/"
PATH_JSON_INFO = "/data1/students/manolina/MedSAM2/LLD-MMRI_MedSAM2/LLD_MMRI_Annotation.json"

# Append MedSAM2 to path
sys.path.append(join(BASE_DIR, 'MedSAM2'))
from sam2.build_sam import build_sam2_video_predictor_npz

# Fix seeds
torch.set_float32_matmul_precision('high')
torch.manual_seed(2024)
torch.cuda.manual_seed(2024)
np.random.seed(2024)

def getLargestCC(segmentation):
    labels = measure.label(segmentation)
    if labels.max() == 0: return segmentation
    return (labels == np.argmax(np.bincount(labels.flat)[1:]) + 1).astype(np.uint8)

def resize_for_medsam(array, size=512):
    d, h, w = array.shape
    out = np.zeros((d, 3, size, size))
    for i in range(d):
        img = Image.fromarray(array[i].astype(np.uint8)).convert("RGB").resize((size, size))
        out[i] = np.array(img).transpose(2, 0, 1)
    return out

if __name__ == "__main__":
    os.makedirs(PRED_SAVE_DIR, exist_ok=True)
    
    with open(PATH_JSON_INFO, 'r') as f:
        annotation_info = json.load(f)['Annotation_info']

    # Initialize model
    original_cwd = os.getcwd()
    os.chdir(join(BASE_DIR, 'MedSAM2'))
    predictor = build_sam2_video_predictor_npz(MODEL_CFG, CHECKPOINT)
    os.chdir(original_cwd)

    # Find images (handle both .nii and .nii.gz)
    all_img_files = sorted(glob(join(IMGS_PATH, '*.nii')) + glob(join(IMGS_PATH, '*.nii.gz')))
    print(f'🚀 Found {len(all_img_files)} images to process in {IMGS_PATH}.\n')

    phase_map = {
        'InPhase': 'In Phase', 'OutPhase': 'Out Phase', 
        'C+A': 'C+A', 'C+V': 'C+V', 'C+Delay': 'C+Delay', 
        'C-pre': 'C-pre', 'DWI': 'DWI', 'T2WI': 'T2WI'
    }

    for img_path in all_img_files:
        img_fname = basename(img_path)
        base_prefix = img_fname.replace('.nii.gz', '').replace('.nii', '')
        
        # Parse pid and phase
        match = re.match(r'(MR-?\d+|MR\d+)_(\d+)_(.+)_0000', base_prefix)
        if not match: 
            print(f"⚠️ Skipping {img_fname}, could not parse format.")
            continue
        pid, lid, phase_str = match.groups()
        
        if pid not in annotation_info: 
            print(f"⚠️ Skipping {img_fname}, pid {pid} not in JSON.")
            continue
            
        record = next((r for r in annotation_info[pid] if r['phase'] == phase_map.get(phase_str, phase_str)), None)
        if not record: 
            print(f"⚠️ Skipping {img_fname}, phase {phase_str} not in JSON for pid {pid}.")
            continue
        
        # Extract best bounding box
        best_box = max(record['annotation']['lesion']['0']['bbox']['2D_box'], key=lambda x: x['area'])
        bbox = np.array([best_box['x_min'], best_box['y_min'], best_box['x_max'], best_box['y_max']])
        key_idx = int(best_box['slice_idx'])
        
        try:
            sitk_img = sitk.ReadImage(img_path)
            nii_data = sitk.GetArrayFromImage(sitk_img)
            
            # Normalization
            norm = ((np.clip(nii_data, np.percentile(nii_data, 1), np.percentile(nii_data, 99)) - np.percentile(nii_data, 1)) / (np.percentile(nii_data, 99) - np.percentile(nii_data, 1)) * 255).astype(np.uint8)
            
            img_tensor = torch.from_numpy(resize_for_medsam(norm) / 255.0).float().cuda()
            
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                state = predictor.init_state(img_tensor, norm.shape[1], norm.shape[2])
                _, _, logits = predictor.add_new_points_or_box(state, frame_idx=key_idx, obj_id=1, box=bbox)
                
                segs = np.zeros(nii_data.shape, dtype=np.uint8)
                
                # 1. FORWARD PROPAGATION
                for f_idx, _, l in predictor.propagate_in_video(state):
                    segs[f_idx] = (l[0] > 0.0).cpu().numpy()
                    
                # 2. BACKWARD PROPAGATION
                for f_idx, _, l in predictor.propagate_in_video(state, reverse=True):
                    segs[f_idx] = (l[0] > 0.0).cpu().numpy()
                    
                predictor.reset_state(state)
                
            segs = getLargestCC(segs)
            
            # Save NIfTI mask with the same naming convention
            save_seg_name = img_fname.replace('.nii.gz', f'_k{key_idx}_mask.nii.gz').replace('.nii', f'_k{key_idx}_mask.nii')
            sitk_mask = sitk.GetImageFromArray(segs)
            sitk_mask.CopyInformation(sitk_img)
            
            sitk.WriteImage(sitk_mask, join(PRED_SAVE_DIR, save_seg_name))
            
            print(f"✅ Generated & Saved Bidirectional Mask: {save_seg_name}")
            
            # Clear memory
            del img_tensor, state
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"⚠️ Error processing {img_fname}: {e}")

    print("\n" + "="*50)
    print("🎉 Mask Generation Complete!")
    print(f"📍 Bidirectional Masks saved to: {PRED_SAVE_DIR}")
    print("="*50)
