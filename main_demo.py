import cv2 as cv
import numpy as np
import random
import matplotlib.pyplot as plt
import torch
from torch import nn
from torchvision.transforms.functional import to_pil_image
from torchvision.io.image import read_image
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchvision.models import resnet18
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask

from segment_anything import sam_model_registry, SamPredictor


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

set_seed(3407)

# Step 1: read image
img = read_image("SVS_2_LS43_to_LS44_RS6_0000_0_281_2816_3840.jpg")
input_tensor = normalize(resize(img, (224, 224)) / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

# Step 2: load model
model = resnet18(pretrained=True)  # load a pretrained resnet18 model
# model.fc = nn.Linear(512, 2)
# model.load_state_dict(torch.load('crack_resnet18_model.pth')) # Optional:load a trained crack classification model to improve performance
model.eval()  
cam_extractor = GradCAM(model, 'layer4')
out = model(input_tensor.unsqueeze(0))

# Step 3:obtain activation map
activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)

# Step 4: Ensure the correct normalization of the heatmap
heatmap = activation_map[0].squeeze(0).detach().numpy()

# smooth the heatmap values to avoid too dispersed sampling probability, enhance the sampling probability of the high temperature area
heatmap = np.clip(heatmap, 0, None)
heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

# Step 5: resize the heatmap to the original image size
heatmap_resized = cv.resize(heatmap, (img.shape[2], img.shape[1]))

# Step 6: ensure that the high temperature area has a higher sampling probability
def sample_points(heatmap, num_points):
    height, width = heatmap.shape
    flat_heatmap = heatmap.flatten()
    flat_heatmap = np.power(flat_heatmap, 30)  # Raise the weight of the heatmap values through exponentiation to make the high temperature area more significant

    probabilities = flat_heatmap / flat_heatmap.sum()  # Normalize to a probability distribution
    indices = np.arange(len(flat_heatmap))

    sampled_indices = np.random.choice(indices, size=num_points, replace=False, p=probabilities)
    sampled_coords = [(index % width, index // width) for index in sampled_indices]

    return sampled_coords

# Step 7: get sampled points
sampled_points = sample_points(heatmap_resized, num_points=3)

# Step 8: load the pre-trained SAM model
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)

# Step 9: Load the image and prepare it for input to SAM
image = to_pil_image(img)
predictor.set_image(np.array(image))

input_points = np.array(sampled_points)
input_labels = np.ones(input_points.shape[0])  # mark all points as foreground (label=1)

# Step 10: Use the SAM model for segmentation
masks, scores, _ = predictor.predict(
    point_coords=input_points,  
    point_labels=input_labels,  
    multimask_output=False)


# Step 12: Post-process the mask generated by SAM
def post_process_mask(mask):
    # Convert mask to uint8 for OpenCV operations
    mask = (mask * 255).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    return mask

processed_mask = post_process_mask(masks[0])

    
# Step 12: visulize the results
def show_points(coords, labels, ax, marker_size=150):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    
plt.figure(figsize=(10, 50))

# original image
plt.subplot(1, 5, 1)
plt.imshow(image)
plt.title('Original')

# Resize the CAM and overlay it
result = overlay_mask(to_pil_image(img), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)
plt.subplot(1, 5, 2)
plt.imshow(result)
plt.title('CAM Overlay')
plt.axis('off')

# sampled points
plt.subplot(1, 5, 3)
plt.imshow(image)
show_points(input_points, input_labels, plt.gca())
plt.axis('off')
plt.title('Pointed')    


# SAM Segmented
plt.subplot(1, 5, 4)
plt.imshow(image)
plt.imshow(masks[0], cmap='jet', alpha=0.5)
plt.axis('off')
plt.title('SAM Segmented')

# Post-Processed SAM Segmented
plt.subplot(1, 5, 5)
plt.imshow(image)
plt.imshow(processed_mask, cmap='jet', alpha=0.5)
plt.axis('off')
plt.title('Post-Processed')
plt.show()