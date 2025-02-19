import os
from glob import glob
import numpy as np
import random
import cv2 as cv
from PIL import Image
import torch
from torchvision.transforms.functional import to_pil_image, normalize, resize
from torchvision.models import resnet18
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
from segment_anything import sam_model_registry, SamPredictor
import matplotlib.pyplot as plt


# Set random seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(3407)


# Get image paths from a folder
def get_image_paths(folder_path, extensions=["jpg", "png", "jpeg", "tif"]):
    image_paths = [r"D:\Keele\CSC40040毕业设计\Project\evalu\DIC_crack_dataset\Test"]
    for ext in extensions:
        image_paths.extend(glob(os.path.join(folder_path, f"*.{ext}")))
    return image_paths


# Read image with support for .tif files
def read_image(image_path):
    img = Image.open(image_path).convert("RGB")  # Convert all formats to RGB
    return torch.tensor(np.array(img)).permute(2, 0, 1)  # Convert to PyTorch tensor (C, H, W)


# Sample points based on heatmap
def sample_points(heatmap, num_points):
    height, width = heatmap.shape
    flat_heatmap = heatmap.flatten()
    flat_heatmap = np.power(flat_heatmap, 30)  # Amplify high values
    probabilities = flat_heatmap / flat_heatmap.sum()
    indices = np.arange(len(flat_heatmap))
    sampled_indices = np.random.choice(indices, size=num_points, replace=False, p=probabilities)
    sampled_coords = [(index % width, index // width) for index in sampled_indices]
    return np.array(sampled_coords)


# Post-process mask with morphological operations
def post_process_mask(mask):
    mask = (mask * 255).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    return mask


# Save visualization results
def save_visualization(image, heatmap, sampled_points, mask, processed_mask, save_path):
    plt.figure(figsize=(10, 50))

    # Original image
    plt.subplot(1, 5, 1)
    plt.imshow(to_pil_image(image))
    plt.title('Original')

    # Heatmap overlay
    result = overlay_mask(to_pil_image(image), to_pil_image(heatmap, mode='F'), alpha=0.5)
    plt.subplot(1, 5, 2)
    plt.imshow(result)
    plt.title('CAM Overlay')
    plt.axis('off')

    # Sampled points
    plt.subplot(1, 5, 3)
    plt.imshow(to_pil_image(image))
    plt.scatter(sampled_points[:, 0], sampled_points[:, 1], color='red', marker='o')
    plt.title('Sampled Points')
    plt.axis('off')

    # Segmented mask
    plt.subplot(1, 5, 4)
    plt.imshow(to_pil_image(image))
    plt.imshow(mask, cmap='jet', alpha=0.5)
    plt.title('SAM Segmented')
    plt.axis('off')

    # Post-processed mask
    plt.subplot(1, 5, 5)
    plt.imshow(to_pil_image(image))
    plt.imshow(processed_mask, cmap='jet', alpha=0.5)
    plt.title('Post-Processed')
    plt.axis('off')

    plt.savefig(save_path)
    plt.close()


# Process a single image
def process_single_image(image_path, model, cam_extractor, predictor, num_points=3):
    print(f"Processing: {image_path}")

    # Load and preprocess image
    img = read_image(image_path)
    input_tensor = normalize(resize(img.float() / 255., [224, 224]), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    # Generate Grad-CAM heatmap
    out = model(input_tensor.unsqueeze(0))
    activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)
    heatmap = activation_map[0].squeeze(0).detach().numpy()
    heatmap = np.clip(heatmap, 0, None)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    heatmap_resized = cv.resize(heatmap, (img.shape[2], img.shape[1]))

    # Sample points
    sampled_points = sample_points(heatmap_resized, num_points)

    # Perform SAM segmentation
    predictor.set_image(img.permute(1, 2, 0).numpy())  # Convert to HWC
    masks, scores, _ = predictor.predict(
        point_coords=np.array(sampled_points),
        point_labels=np.ones(len(sampled_points)),
        multimask_output=False
    )

    # Post-process mask
    processed_mask = post_process_mask(masks[0])

    return img, heatmap_resized, sampled_points, masks[0], processed_mask


# Process all images in a folder
def process_images_in_folder(folder_path, save_dir, num_points=3, model_type="vit_h", sam_checkpoint="sam_vit_h_4b8939.pth", device="cuda"):
    os.makedirs(save_dir, exist_ok=True)
    image_paths = get_image_paths(folder_path)
    if not image_paths:
        print(f"No images found in folder: {folder_path}")
        return

    print(f"Found {len(image_paths)} images in {folder_path}")

    # Load models
    model = resnet18(pretrained=True)
    model.eval()
    cam_extractor = GradCAM(model, 'layer4')

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    # Process each image
    for image_path in image_paths:
        try:
            img, heatmap, sampled_points, mask, processed_mask = process_single_image(
                image_path, model, cam_extractor, predictor, num_points
            )
            save_path = os.path.join(save_dir, os.path.basename(image_path).replace(".tif", ".jpg"))
            save_visualization(img, heatmap, sampled_points, mask, processed_mask, save_path)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

# executive project
process_images_in_folder(
    folder_path="D:\Keele\CSC40040毕业设计\Project\evalu\DIC_crack_dataset\Test",
    save_dir="results",
    num_points=3,
    model_type="vit_h",
    sam_checkpoint="sam_vit_h_4b8939.pth",
    device="cuda"
)
