import os
import json
import re
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from glob import glob

def show_mask(mask, ax, mask_color):
    """Overlays a mask with the given color on the given axes."""
    color = np.concatenate([mask_color, np.array([0.5])], axis=0)
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def draw_bbox(ax, bbox):
    """Draws a red bounding box on the given axes."""
    # bbox format is [x_min, y_min, x_max, y_max]
    x_min, y_min, x_max, y_max = bbox
    width = x_max - x_min
    height = y_max - y_min
    
    rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='red', facecolor='none')
    ax.add_patch(rect)

def calculate_dice(pred, target):
    """Calculates the Dice score between two binary arrays."""
    p, t = np.asarray(pred).astype(bool), np.asarray(target).astype(bool)
    if p.sum() == 0 and t.sum() == 0:
        return 1.0
    return 2.0 * np.logical_and(p, t).sum() / (p.sum() + t.sum())

def main():
    # Directories (Update these if necessary based on your server setup)
    images_dir = "/data1/students/manolina/MedSAM2/50_image/"
    labels_dir = "/data1/students/manolina/MedSAM2/50_label/"
    masks_dir = "/data1/students/manolina/50_generated_mask/"
    json_path = "/data1/students/manolina/MedSAM2/LLD-MMRI_MedSAM2/LLD_MMRI_Annotation.json"
    output_dir = "/data1/students/manolina/50_comparisons_new/"

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load JSON Annotation file
    with open(json_path, 'r') as f:
        annotation_info = json.load(f)['Annotation_info']

    # Map phases as done in the evaluation script
    phase_map = {
        'InPhase': 'In Phase', 'OutPhase': 'Out Phase', 
        'C+A': 'C+A', 'C+V': 'C+V', 'C+Delay': 'C+Delay', 
        'C-pre': 'C-pre', 'DWI': 'DWI', 'T2WI': 'T2WI'
    }

    # Find all 50 images
    image_files = glob(os.path.join(images_dir, "*.nii")) + glob(os.path.join(images_dir, "*.nii.gz"))
    print(f"Found {len(image_files)} images to process.")

    for img_path in image_files:
        nii_fname = os.path.basename(img_path)
        base_prefix = nii_fname.replace('.nii.gz', '').replace('.nii', '')
        
        # 1. Parse filename to find pid and phase
        match = re.match(r'(MR-?\d+|MR\d+)_(\d+)_(.+)_0000', base_prefix)
        if not match:
            print(f"⚠️ Warning: Could not parse filename format for {nii_fname}")
            continue
        pid, lid, phase_str = match.groups()

        # 2. Resolve Paths for Ground Truth and Generated Mask
        gt_filename = nii_fname.replace('_0000.nii', '.nii').replace('_0000.nii.gz', '.nii.gz')
        gt_path = os.path.join(labels_dir, gt_filename)
        if not os.path.exists(gt_path):
            gt_path = os.path.join(labels_dir, nii_fname)
            
        if not os.path.exists(gt_path):
            print(f"⚠️ Warning: Ground truth not found for {nii_fname}")
            continue

        search_pattern = os.path.join(masks_dir, f"{base_prefix}_k*_mask.nii*")
        matching_masks = glob(search_pattern)
        if not matching_masks:
            print(f"⚠️ Warning: Generated mask not found for {nii_fname}")
            continue
        mask_path = matching_masks[0]

        # 3. JSON Lookup for Key Slice and Bounding Box
        if pid not in annotation_info:
            print(f"⚠️ Warning: Patient {pid} not found in JSON.")
            continue
            
        record = next((r for r in annotation_info[pid] if r['phase'] == phase_map.get(phase_str, phase_str)), None)
        if not record:
            print(f"⚠️ Warning: Phase {phase_str} not found in JSON for patient {pid}.")
            continue

        # Extract the bounding box with the maximum area
        best_box = max(record['annotation']['lesion']['0']['bbox']['2D_box'], key=lambda x: x['area'])
        bbox = [best_box['x_min'], best_box['y_min'], best_box['x_max'], best_box['y_max']]
        key_idx = int(best_box['slice_idx'])

        # 4. Load NIfTI Data
        try:
            nii_data = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
            gt_data = (sitk.GetArrayFromImage(sitk.ReadImage(gt_path)) > 0).astype(np.uint8)
            segs_data = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
        except Exception as e:
            print(f"⚠️ Error reading NIfTI files for {nii_fname}: {e}")
            continue

        # Normalize the input image array visually (matching your eval script)
        norm = ((np.clip(nii_data, np.percentile(nii_data, 1), np.percentile(nii_data, 99)) - np.percentile(nii_data, 1)) / (np.percentile(nii_data, 99) - np.percentile(nii_data, 1)) * 255).astype(np.uint8)

        max_slices = norm.shape[0]

        # 5. Generate Multi-Slice Plots (x-2, x-1, x, x+1, x+2)
        slices_to_plot = [key_idx - 2, key_idx - 1, key_idx, key_idx + 1, key_idx + 2]
        
        # Create a single figure with 5 rows and 3 columns
        fig, axes = plt.subplots(5, 3, figsize=(18, 30))
        
        for i, idx in enumerate(slices_to_plot):
            # If slice is out of bounds, just hide the axes
            if idx < 0 or idx >= max_slices:
                axes[i, 0].axis('off')
                axes[i, 1].axis('off')
                axes[i, 2].axis('off')
                continue

            # Panel 1: Input Image
            axes[i, 0].imshow(norm[idx], cmap='gray')
            axes[i, 0].set_title(f'Input Image (Slice {idx})', fontsize=16)
            axes[i, 0].axis('off')

            # Panel 2: Ground Truth + Red Prompt Box
            axes[i, 1].imshow(norm[idx], cmap='gray')
            show_mask(gt_data[idx], ax=axes[i, 1], mask_color=np.array([0, 1, 0]))
            draw_bbox(axes[i, 1], bbox)
            axes[i, 1].set_title('Ground Truth (Green) + Prompt (Red)', fontsize=16)
            axes[i, 1].axis('off')

            # Panel 3: Generated Mask + Red Prompt Box
            axes[i, 2].imshow(norm[idx], cmap='gray')
            show_mask(segs_data[idx], ax=axes[i, 2], mask_color=np.array([251/255, 252/255, 30/255]))
            draw_bbox(axes[i, 2], bbox)
            
            # Calculate and display Dice score for this specific slice
            slice_dice = calculate_dice(segs_data[idx], gt_data[idx])
            axes[i, 2].set_title(f'Generated Mask (Yellow) + Prompt (Red)\nDice Score: {slice_dice:.4f}', fontsize=16)
            axes[i, 2].axis('off')

        plt.tight_layout()
        
        # Save format: {base_prefix}_5_slice_comparison.png
        save_name = f"{base_prefix}_5_slice_comparison.png"
        plt.savefig(os.path.join(output_dir, save_name), bbox_inches='tight')
        plt.close(fig) # prevent memory leaks

        print(f"Generated stacked comparison for {nii_fname} (key slice {key_idx})")

    print(f"\n✅ All visualizations complete! Saved to {output_dir}")

if __name__ == "__main__":
    main()
