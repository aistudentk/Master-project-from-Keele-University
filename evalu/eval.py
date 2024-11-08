import cv2 as cv
import numpy as np
import os
import random
import copy
import torch
from torch import nn
from torchvision.io.image import read_image
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchvision.io.image import read_image
from torchvision.models import resnet18
from torchcam.methods import CAM, GradCAM, SmoothGradCAMpp, SSCAM, LayerCAM, XGradCAM, ScoreCAM, ISCAM
from segment_anything import sam_model_registry, SamPredictor
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score
from PIL import Image
import torchvision.transforms as transforms
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--cam_method', type=str, default='gradcam', choices=['cam', 'gradcam', 'xgradcam', 'layercam', 'smoothgradcam', 'scorecam'])
parser.add_argument('--model_weight', type=str, default='Imagenet_pretrained', choices=['Imagenet_pretrained', 'crack_cls'])
parser.add_argument('--num_points', type=int, default=10)
parser.add_argument('--weight_exponent', type=int, default=1)
parser.add_argument('--seed', type=int, default=3407)


# Utility functions from eval.py
def calculate_metrics(predicted, ground_truth):
    # Binarize predicted and ground truth masks
    predicted = (predicted > 0).astype(np.uint8)
    ground_truth = (ground_truth > 0).astype(np.uint8)

    predicted_flat = predicted.flatten()
    ground_truth_flat = ground_truth.flatten()

    precision = precision_score(ground_truth_flat, predicted_flat, zero_division=0)
    recall = recall_score(ground_truth_flat, predicted_flat, zero_division=0)
    f1 = f1_score(ground_truth_flat, predicted_flat, zero_division=0)
    iou = jaccard_score(ground_truth_flat, predicted_flat, zero_division=0)

    return {"Precision": precision, "Recall": recall, "F1": f1, "IoU": iou}


# Convert black & white TIFF to RGB
def convert_tif_to_rgb(image_path):
    image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    rgb_image = cv.cvtColor(image, cv.COLOR_GRAY2RGB)  # Convert grayscale to RGB
    return rgb_image


# Function to overlay a mask on the original image
def overlay_mask_on_image(original_image, mask, output_path):
    cv.imwrite(output_path, mask)


def post_process_mask(mask):
    # Ensure mask is processed independently by making a deep copy
    mask_copy = copy.deepcopy(mask)

    # Convert mask to uint8 for OpenCV operations
    mask_copy = (mask_copy * 255).astype(np.uint8)

    # Define the morphological operations kernel
    kernel = np.ones((3, 3), np.uint8)

    # Apply opening and closing to remove small artifacts
    mask_copy = cv.morphologyEx(mask_copy, cv.MORPH_OPEN, kernel)
    mask_copy = cv.morphologyEx(mask_copy, cv.MORPH_CLOSE, kernel)

    # Find contours in the mask
    contours, _ = cv.findContours(mask_copy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Filter the contours based on a min and max area threshold
    min_area = 100
    max_area = 15000

    # Initialize a blank mask to draw the filtered contours
    filtered_image = np.zeros_like(mask_copy)

    for contour in contours:
        area = cv.contourArea(contour)
        # Only draw contours that meet the area criteria
        if min_area < area < max_area:
            cv.drawContours(filtered_image, [contour], -1, 255, thickness=cv.FILLED)

    return filtered_image


# Step 0: set seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Step 1: model loading function
def load_cls_model(model_weight, cam_method='gradcam'):
    cls_model = resnet18(pretrained=True)
    if model_weight == 'crack_cls':
        print('Loading crack_cls model...')
        cls_model.fc = nn.Linear(512, 2)
        cls_model.load_state_dict(torch.load('checkpoints/output/epoch=99-val_loss=0.00.ckpt'))
    cls_model.eval()
    if cam_method == 'gradcam':
        cam_extractor = GradCAM(cls_model, 'layer4')
    elif cam_method == 'cam':
        cam_extractor = CAM(cls_model, 'layer4')
    elif cam_method == 'xgradcam':
        cam_extractor = XGradCAM(cls_model, 'layer4')
    elif cam_method == 'layercam':
        cam_extractor = LayerCAM(cls_model, 'layer4')
    elif cam_method == 'smoothgradcam':
        cam_extractor = SmoothGradCAMpp(cls_model, 'layer4')
    elif cam_method == 'scorecam':
        cam_extractor = ScoreCAM(cls_model, 'layer4')
    return cls_model, cam_extractor


def load_seg_model():
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    return predictor


# Step 2: process a single image with the CAM to SAM zero-shot segmentation method
def process_image(img_path, model, cam_extractor):
    if img_path.endswith('.tif'):
        img = convert_tif_to_rgb(img_path)  # Convert TIFF to RGB if necessary
    else:
        img = read_image(img_path)

    input_tensor = normalize(resize(img, (224, 224)) / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    out = model(input_tensor.unsqueeze(0))
    activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)

    heatmap = activation_map[0].squeeze(0).detach().numpy()
    heatmap = np.clip(heatmap, 0, None)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

    heatmap_resized = cv.resize(heatmap, (img.shape[2], img.shape[1]))

    return heatmap_resized, img


def sample_points(heatmap, num_points, weight_exponent=30):
    height, width = heatmap.shape
    flat_heatmap = heatmap.flatten()
    flat_heatmap = np.power(flat_heatmap, weight_exponent)  # Raise the weight of the heatmap values through exponentiation to make the high temperature area more significant

    probabilities = flat_heatmap / flat_heatmap.sum()  # Normalize to a probability distribution
    indices = np.arange(len(flat_heatmap))

    sampled_indices = np.random.choice(indices, size=num_points, replace=False, p=probabilities)
    sampled_coords = [(index % width, index // width) for index in sampled_indices]

    return sampled_coords


# Step 3: batch processing and evaluation
def evaluate_model_on_test_set(model, seg_predictor, cam_extractor, test_image_folder, gt_folder, output_folder, num_points=3, weight_exponent=30, seed=3407):
    print("Starting evaluation..., num_points: {}, weight_exponent: {}".format(num_points, weight_exponent

                                                                               ))
    results = []
    total_metrics = {"Precision": 0, "Recall": 0, "F1": 0, "IoU": 0}
    num_images = 0

    for image_name in os.listdir(test_image_folder):
        set_seed(seed)
        if image_name.endswith(('.jpg', '.png', '.tif')):
            img_path = os.path.join(test_image_folder, image_name)
            # Append '_mask' to the image name for ground truth
            gt_image_name = image_name.replace('.jpg', '_mask.png')
            gt_path = os.path.join(gt_folder, gt_image_name)

            # Process the image with CAM and SAM
            heatmap_resized, original_img = process_image(img_path, model, cam_extractor)
            sampled_points = sample_points(heatmap_resized, num_points, weight_exponent)
            # print(sampled_points)

            # img = read_image(img_path)
            image = to_pil_image(original_img)
            seg_predictor.set_image(np.array(image))

            input_points = np.array(sampled_points)
            input_labels = np.ones(input_points.shape[0])  # mark all points as foreground (label=1)

            # Step 10: Use the SAM model for segmentation
            predicted_mask, scores, _ = seg_predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=False
            )

            foreground_count = np.sum(predicted_mask)
            background_count = np.size(predicted_mask) - foreground_count
            if foreground_count > background_count:
                predicted_mask = np.logical_not(predicted_mask)

            predicted_mask = post_process_mask(predicted_mask[0])  # Apply post-processing to the mask

            # Load ground truth as grayscale
            ground_truth = cv.imread(gt_path, cv.IMREAD_GRAYSCALE)
            # Calculate metrics
            metrics = calculate_metrics(predicted_mask, ground_truth)
            results.append((image_name, metrics))
            # print(f"Processed {image_name}: {metrics}")

            # Add metrics to total
            for key in total_metrics:
                total_metrics[key] += metrics[key]
            num_images += 1
            print(f"Processed {num_images} images")

            # Save the result image with overlay
            output_image_path = os.path.join(output_folder, f"{image_name.replace('.tif', ' ')}_result.png")
            overlay_mask_on_image(original_img, predicted_mask, output_image_path)

    # Compute average metrics
    average_metrics = {key: total_metrics[key] / num_images for key in total_metrics}

    return results, average_metrics


def write_to_json(result_entry, output_json='experiment_result.json'):
    if os.path.exists(output_json):
        with open(output_json, 'r', encoding='utf-8') as file:
            results_data = json.load(file)
    else:
        results_data = []

    # Add new result to the list
    results_data.append(result_entry)

    # Save back the updated data
    with open(output_json, 'w', encoding='utf-8') as file:
        json.dump(results_data, file, indent=4, ensure_ascii=False)


def convert_all_tif_to_jpg(folder_path):
    if not os.path.exists(folder_path):
        print(f"folder {folder_path} not exists")
        return
    output_folder = os.path.join(folder_path, "test_jpg")
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(folder_path):
        if filename.endswith(".tif") or filename.endswith(".tiff"):
            tif_path = os.path.join(folder_path, filename)
            tif_image = Image.open(tif_path).convert('L')

            rgb_image = tif_image.convert('RGB')

            jpg_filename = os.path.splitext(filename)[0] + ".jpg"
            jpg_path = os.path.join(output_folder, jpg_filename)
            rgb_image.save(jpg_path, 'JPEG')
            print(f"changed {filename} to {jpg_filename}")
        else:
            print(f"skipped {filename} because it is not a tif file")

    print("all done!")


if __name__ == "__main__":
    args = parser.parse_args()
    set_seed(args.seed)

    TEST_IMAGE_FOLDER = "DIC_crack_dataset/test_jpg"
    GROUND_TRUTH_FOLDER = "DIC_crack_dataset/test_GT"
    OUTPUT_FOLDER = "test_results_single"

    # Ensure the output folder exists
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    # Load model
    cls_model, cam_extractor = load_cls_model(args.model_weight, args.cam_method)
    seg_predictor = load_seg_model()

    # Evaluate model
    results, average_metrics = evaluate_model_on_test_set(cls_model, seg_predictor, cam_extractor, TEST_IMAGE_FOLDER, GROUND_TRUTH_FOLDER, OUTPUT_FOLDER, args.num_points, args.weight_exponent,
                                                          args.seed)

    result_entry = {
        'model': 'resnet18',
        'weight': args.model_weight,
        'cam_method': args.cam_method,
        'seed': args.seed,
        'num_points': args.num_points,
        'weight_exponent': args.weight_exponent,
        'F1': average_metrics['F1'],
        'Precision': average_metrics['Precision'],
        'Recall': average_metrics['Recall'],
        'IoU': average_metrics['IoU']
    }
    write_to_json(result_entry)
    print(f"Average Metrics: F1: {average_metrics['F1']}, Precision: {average_metrics['Precision']}, Recall: {average_metrics['Recall']}, IoU: {average_metrics['IoU']}")
