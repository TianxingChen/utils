import numpy as np
import torch
import matplotlib.pyplot as plt
import pdb
import cv2
import sys
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
from os.path import dirname, abspath, join
import os
import sys
import random
from PIL import Image

class AutoGenerateMask:
    def __init__(self, check_point_path="unknown", model_type = "vit_h", device="cuda:0"):
        assert check_point_path != "unknown", 'unknown SAM checkpoint path'
        sam_checkpoint = check_point_path + f"/sam_{model_type}.pth"
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.to(device=device)
        # self.predictor = SamPredictor(sam)
        self.auto_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=64,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,  # Requires open-cv to run post-processing
        )
    
    def load_image_local(self, url):
        image = cv2.imread(url)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def auto_generate_mask(self, image):
        masks = self.auto_generator.generate(image)
        # ['segmentation', 'area', 'bbox', 'predicted_iou', 'point_coords', 'stability_score', 'crop_box']
        '''
            * `segmentation` : the mask
            * `area` : the area of the mask in pixels
            * `bbox` : the boundary box of the mask in XYWH format
            * `predicted_iou` : the model's own prediction for the quality of the mask
            * `point_coords` : the sampled input point that generated this mask
            * `stability_score` : an additional measure of mask quality
            * `crop_box` : the crop of the image used to generate this mask in XYWH format
        '''
        return masks
    
    def get_mask(self, image):
        masks = self.auto_generate_mask(image)
        result_rgb, result_mask = np.zeros_like(image), np.zeros_like(image)
        sorted_masks = sorted(
            masks, 
            key=lambda x: x['predicted_iou'],
            reverse=True
        )
        for i in range(type_num):
            current_color = np.array([random.randint(0, 255) for _ in range(3)])
            result_rgb[sorted_masks[i]['segmentation']] = current_color
            result_mask[sorted_masks[i]['segmentation']] = i+1

        return result_rgb, result_mask
