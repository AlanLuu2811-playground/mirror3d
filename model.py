import sys, os

file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f"{file_dir}/mirror3d/mirror3dnet")

from mirror3d.utils import RefineDepth, unit_vector
from mirror3d_lib.config.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import copy

class M3DWrapper():
    def __init__(
        self,
        cfg_file,
        anchor_normal_npy,
        m3d_weight_path,
        focal_length,
        input_width,
        input_height,
        score_threshold=0.5):

        cfg = get_cfg() 
        cfg.merge_from_file(cfg_file)

        # MODEL
        cfg.ANCHOR_NORMAL_NYP = anchor_normal_npy
        cfg.MODEL.WEIGHTS = m3d_weight_path
        cfg.ANCHOR_NORMAL_CLASS_NUM = np.load(cfg.ANCHOR_NORMAL_NYP).shape[0]

        cfg.FOCAL_LENGTH = focal_length
        cfg.EVAL_HEIGHT = input_height
        cfg.EVAL_WIDTH = input_width
        cfg.INPUT.MIN_SIZE_TEST = input_height
        cfg.INPUT.MAX_SIZE_TEST = input_width
        cfg.REF_MODE = "rawD_border"
        self.input_width = input_width
        self.input_height = input_height
        self.cls_names = ['mirror_glass',]
        self.colors = [[255, 56, 56],]  # bgr format
        self.ref_mode = cfg.REF_MODE

        self.predictor = DefaultPredictor(cfg)
        self.depth_refiner = RefineDepth(cfg.FOCAL_LENGTH, cfg.REF_BORDER_WIDTH, cfg.EVAL_WIDTH, cfg.EVAL_HEIGHT)
        self.score_threshold = score_threshold
        self.anchor_normals = np.load(cfg.ANCHOR_NORMAL_NYP)
        self.warmup_num = 5
        self.warmup(self.warmup_num)

    def resizeWithCropFactor(self, img, size, intrinsic=None):
        h1, w1 = img.shape[:2]
        w2, h2 = size
        if h1 == h2 and w1 == w2:
            return img, intrinsic 
        r1 = w1 / h1
        r2 = w2 / h2
        if r1 >= r2:
            changed_width = int(r2 * h1)
            offset_pxl_pos = int((w1 - changed_width) / 2)
            img = img[:, offset_pxl_pos : offset_pxl_pos + changed_width]
            ratio = h2 / h1
        elif r1 < r2:
            changed_height = int(w1 / r2)
            offset_pxl_pos = int((h1 - changed_height) / 2)
            img = img[offset_pxl_pos : offset_pxl_pos + changed_height, :]
            ratio = w2 / w1

        if intrinsic is not None:
            new_intrinsic = intrinsic.copy()
            new_intrinsic[0, -1] *= ratio
            new_intrinsic[1, -1] *= ratio
            new_intrinsic[0, 0] *= ratio
            new_intrinsic[1, 1] *= ratio
            return cv2.resize(img, (w2, h2)), new_intrinsic

        return cv2.resize(img, (w2, h2)), None

    def warmup(self, num_warmup):
        for i in range(num_warmup):
            dummy_input = torch.rand(3, self.input_height, self.input_width)
            with torch.no_grad():
                _ = self.predictor.model([{"image": dummy_input}])

    def preprocess(self, color_imgs, depth_imgs):
        assert len(color_imgs) == len(depth_imgs)
        input_batch = []
        for color_img, depth_img in zip(color_imgs, depth_imgs):
            ori_h, ori_w, _ = color_img.shape
            input_img, _ = self.resizeWithCropFactor(color_img, (self.input_width, self.input_height))
            input_img = torch.from_numpy(input_img).permute(2, 0, 1)

            input_batch.append({
                "image": input_img,
                "height": ori_h,
                "width": ori_w,
                "ori_depth_img": depth_img,
                "ori_img": color_img
            })

        return input_batch

    def run(self, color_imgs, depth_imgs):
        input_batch = self.preprocess(color_imgs, depth_imgs)
        output_batch = copy.deepcopy(input_batch)
        for output in output_batch:
            output.pop("image")
            output['refined_depth'] = None
            output['pred_bboxes'] = None
            output['pred_cls'] = None
            output['pred_scores'] = None
            output['pred_masks'] = None
            output['pred_anchor_residuals'] = None
            output['pred_anchors_cls'] = None

        with torch.no_grad():
            outputs = self.predictor.model(input_batch)
        
        for output in outputs[0]:
            output["instances"] = output["instances"][output["instances"].scores > self.score_threshold]

        for i, one_output in enumerate(outputs[0]):
            instances = one_output["instances"].to('cpu')
            pred_bboxes = instances.pred_boxes.tensor.detach().numpy()
            pred_cls = instances.pred_classes.detach().numpy()
            pred_scores = instances.scores.detach().numpy()
            pred_masks = instances.pred_masks.detach().numpy()
            pred_anchor_residuals = instances.pred_residuals.detach().numpy()
            pred_anchors_cls = instances.pred_anchor_classes.detach().numpy()

            output_batch[i]['pred_bboxes'] = pred_bboxes
            output_batch[i]['pred_cls'] = pred_cls
            output_batch[i]['pred_scores'] = pred_scores
            output_batch[i]['pred_masks'] = pred_masks
            output_batch[i]['pred_anchor_residuals'] = pred_anchor_residuals
            output_batch[i]['pred_anchors_cls'] = pred_anchors_cls

            if pred_bboxes.shape[0] <= 0:
                return output_batch

            # refine depth
            ori_depth = input_batch[i]['ori_depth_img']
            pred_mask = np.zeros((ori_depth.shape))
            pred_mask = pred_mask.astype(bool)
            ref_depth = ori_depth.copy()

            if pred_masks.shape[0] > 0:
                for index, one_pred_mask in enumerate(pred_masks):
                    one_pred_mask = one_pred_mask.astype(np.int8)
                    one_pred_mask = cv2.resize(one_pred_mask, ori_depth.shape[::-1], interpolation=cv2.INTER_NEAREST)
                    one_pred_mask = one_pred_mask.astype(bool)
                    to_refine_area = one_pred_mask
                    to_refine_area = np.logical_and(pred_mask==False, to_refine_area)
                    if to_refine_area.sum() == 0:
                        continue
                    pred_mask = np.logical_or(pred_mask , one_pred_mask)
                    if pred_anchors_cls[index] >= self.anchor_normals.shape[0]:
                        continue
                    pred_normal = self.anchor_normals[pred_anchors_cls[index]] +  pred_anchor_residuals[index]
                    pred_normal = unit_vector(pred_normal)

                    if "border" in self.ref_mode:
                        ref_depth = self.depth_refiner.refine_depth_by_mirror_border(
                            one_pred_mask.squeeze(), 
                            pred_normal, 
                            ref_depth,
                            reduce_half=True)
                    else:
                        raise NotImplementedError("Only border-based refine mode is implemented for now.")

            ref_depth[ref_depth < 0] = 0
            output_batch[i]['refined_depth'] = ref_depth

        return output_batch

    def draw_segmentation(
        self,
        image,
        mask,
        box,
        pred_cls,
        score,
        color=None,
        alpha=0.5,
        scale=1.0):
        """
        Input:
            image: HxWx3 (BGR)
            mask: HxW (bool or 0/1)
            box: [x1, y1, x2, y2]
            pred_cls: int
            score: float
            color: tuple or list of 3 ints (optional)
            alpha: float (optional)
            scale: float (optional)
        """

        # --- Resize image ---
        if scale != 1.0:
            new_w = int(image.shape[1] * scale)
            new_h = int(image.shape[0] * scale)
            image = cv2.resize(image, (new_w, new_h))

        img = image.copy()

        # Random color if not provided
        if color is None:
            color = np.random.randint(0, 255, size=3).tolist()

        # --- Resize mask ---
        if mask is not None:
            if scale != 1.0:
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (img.shape[1], img.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )
            mask = mask.astype(bool)

            colored_mask = np.zeros_like(img, dtype=np.uint8)
            colored_mask[mask] = color
            img = cv2.addWeighted(img, 1.0, colored_mask, alpha, 0)

        # --- Resize box ---
        x1, y1, x2, y2 = box
        if scale != 1.0:
            x1 = int(x1 * scale)
            y1 = int(y1 * scale)
            x2 = int(x2 * scale)
            y2 = int(y2 * scale)

        # --- Draw bounding box ---
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # --- Label ---
        label = f"{self.cls_names[pred_cls]}: {score:.2f}"

        (w, h), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            1
        )

        # Prevent label going outside image
        y_text = max(y1, h + baseline)

        # --- Draw label background ---
        cv2.rectangle(
            img,
            (x1, y_text - h - baseline),
            (x1 + w, y_text),
            color,
            -1
        )

        # --- Put text ---
        cv2.putText(
            img,
            label,
            (x1, y_text - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        return img