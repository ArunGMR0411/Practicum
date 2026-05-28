"""Shared helpers for Face Anonymization Made Simple backend execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class FAMSDetection:
    landmarks: np.ndarray
    bbox: tuple[int, int, int, int]
    image_to_face_mat: np.ndarray
    face_image: Image.Image


def landmarks_bbox(landmarks: np.ndarray) -> tuple[int, int, int, int]:
    xs = landmarks[:, 0]
    ys = landmarks[:, 1]
    return (
        int(np.floor(xs.min())),
        int(np.floor(ys.min())),
        int(np.ceil(xs.max())),
        int(np.ceil(ys.max())),
    )


def bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(1, ax2 - ax1) * max(1, ay2 - ay1)
    area_b = max(1, bx2 - bx1) * max(1, by2 - by1)
    return inter / float(area_a + area_b - inter)


def select_detections_for_boxes(
    detections: list[FAMSDetection],
    reviewed_boxes: list[tuple[int, int, int, int]],
    overlap_iou_threshold: float,
) -> list[FAMSDetection]:
    selected: list[FAMSDetection] = []
    for detection in detections:
        if any(bbox_iou(detection.bbox, reviewed) >= overlap_iou_threshold for reviewed in reviewed_boxes):
            selected.append(detection)
    return selected


def build_fams_detections(
    image: Image.Image,
    landmarks_list: list[np.ndarray] | None,
    *,
    get_transform_mat: object,
    face_image_size: int,
    face_type: object,
) -> list[FAMSDetection]:
    if not landmarks_list:
        return []
    array = np.array(image)[:, :, :3]
    detections: list[FAMSDetection] = []
    for landmarks in landmarks_list:
        landmarks_np = np.array(landmarks)
        image_to_face_mat = get_transform_mat(landmarks_np, face_image_size, face_type)
        face_array = cv2.warpAffine(
            array,
            image_to_face_mat,
            (face_image_size, face_image_size),
            cv2.INTER_LANCZOS4,
            borderValue=(255, 255, 255),
        )
        detections.append(
            FAMSDetection(
                landmarks=landmarks_np,
                bbox=landmarks_bbox(landmarks_np),
                image_to_face_mat=image_to_face_mat,
                face_image=Image.fromarray(face_array),
            )
        )
    return detections


def build_fams_detections_from_reviewed_boxes(
    image: Image.Image,
    reviewed_boxes: list[tuple[int, int, int, int]],
    *,
    face_detector: object,
    get_transform_mat: object,
    face_image_size: int,
    face_type: object,
    crop_padding_ratio: float = 0.35,
) -> list[FAMSDetection]:
    """Build FAMS detections by running landmarks only inside reviewed face ROIs.

    This avoids full-frame face-alignment detection on large 4K egocentric images,
    which is the main VRAM failure mode on constrained GPUs.
    """
    if not reviewed_boxes:
        return []

    array = np.array(image)[:, :, :3]
    image_h, image_w = array.shape[:2]
    detections: list[FAMSDetection] = []

    for reviewed_box in reviewed_boxes:
        x1, y1, x2, y2 = reviewed_box
        box_w = max(1, x2 - x1)
        box_h = max(1, y2 - y1)
        pad_x = int(round(box_w * crop_padding_ratio))
        pad_y = int(round(box_h * crop_padding_ratio))
        crop_left = max(0, x1 - pad_x)
        crop_top = max(0, y1 - pad_y)
        crop_right = min(image_w, x2 + pad_x)
        crop_bottom = min(image_h, y2 + pad_y)
        crop_array = array[crop_top:crop_bottom, crop_left:crop_right]
        if crop_array.size == 0:
            continue

        landmarks_list = face_detector.get_landmarks(crop_array)
        if not landmarks_list:
            continue

        lifted_candidates: list[np.ndarray] = []
        for landmarks in landmarks_list:
            landmarks_np = np.array(landmarks, dtype=np.float32)
            landmarks_np[:, 0] += float(crop_left)
            landmarks_np[:, 1] += float(crop_top)
            lifted_candidates.append(landmarks_np)

        best_landmarks = max(
            lifted_candidates,
            key=lambda item: bbox_iou(landmarks_bbox(item), reviewed_box),
        )
        image_to_face_mat = get_transform_mat(best_landmarks, face_image_size, face_type)
        face_array = cv2.warpAffine(
            array,
            image_to_face_mat,
            (face_image_size, face_image_size),
            cv2.INTER_LANCZOS4,
            borderValue=(255, 255, 255),
        )
        detections.append(
            FAMSDetection(
                landmarks=best_landmarks,
                bbox=landmarks_bbox(best_landmarks),
                image_to_face_mat=image_to_face_mat,
                face_image=Image.fromarray(face_array),
            )
        )

    return detections
