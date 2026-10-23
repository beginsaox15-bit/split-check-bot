"""Локальная геометрия и щадящая обработка чека; пиксели не сохраняются."""

from io import BytesIO

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from bot.services.receipt_models import UnreadableReceipt


cv2.setNumThreads(1)
MAX_SIDE = 3600


def rectify_receipt(gray: np.ndarray) -> tuple[np.ndarray, bool]:
    """Разворачивает только крупный выпуклый четырёхугольник светлой бумаги."""
    scale = min(1.0, 1200 / max(gray.shape))
    small = cv2.resize(gray, None, fx=scale, fy=scale) if scale < 1 else gray
    blurred = cv2.GaussianBlur(small, (5, 5), 0)
    edges = cv2.morphologyEx(cv2.Canny(blurred, 40, 120), cv2.MORPH_CLOSE,
                             np.ones((5, 5), np.uint8))
    _, light = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours = []
    for mask in (edges, light):
        contours.extend(cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0])
    area = small.size
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        if not .20 * area <= cv2.contourArea(contour) <= .97 * area:
            continue
        polygon = cv2.approxPolyDP(contour, .025 * cv2.arcLength(contour, True), True)
        if len(polygon) != 4 or not cv2.isContourConvex(polygon):
            continue
        points = polygon.reshape(4, 2).astype(np.float32)
        sums, differences = points.sum(axis=1), np.diff(points, axis=1).ravel()
        ordered = points[[sums.argmin(), differences.argmin(), sums.argmax(), differences.argmax()]]
        if len(np.unique(ordered, axis=0)) != 4:
            continue
        mask = np.zeros_like(small)
        cv2.fillConvexPoly(mask, polygon, 255)
        inside, outside = small[mask > 0], small[mask == 0]
        if inside.mean() < 120 or not outside.size or inside.mean() - outside.mean() < 15:
            continue
        corners = ordered / scale
        # Запас защищает крайние буквы от обрезки по контуру бумаги.
        center = corners.mean(axis=0)
        corners = center + (corners - center) * 1.015
        corners[:, 0] = np.clip(corners[:, 0], 0, gray.shape[1] - 1)
        corners[:, 1] = np.clip(corners[:, 1], 0, gray.shape[0] - 1)
        tl, tr, br, bl = corners
        width = round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
        height = round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
        if min(width, height) < 80 or max(width, height) / min(width, height) > 12:
            continue
        destination = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
        transform = cv2.getPerspectiveTransform(corners.astype(np.float32), destination)
        return cv2.warpPerspective(gray, transform, (width, height), borderValue=255), True
    return gray, False


def deskew(gray: np.ndarray) -> tuple[np.ndarray, float]:
    """Ищет небольшой наклон по горизонтальным проекциям, без обрезки букв."""
    ratio = min(1.0, 800 / max(gray.shape))
    small = cv2.resize(gray, None, fx=ratio, fy=ratio) if ratio < 1 else gray
    _, mask = cv2.threshold(small, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if mask.mean() < .002 or mask.mean() > .35:
        return gray, 0.0
    h, w = mask.shape

    def projection_score(angle):
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
        rotated = cv2.warpAffine(mask, matrix, (w, h), flags=cv2.INTER_NEAREST)
        return float(np.var(rotated.sum(axis=1)))

    baseline = projection_score(0)
    score, angle = max((projection_score(angle), angle) for angle in range(-12, 13))
    if not angle or score < baseline * 1.12:
        return gray, 0.0
    h, w = gray.shape
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    cosine, sine = abs(matrix[0, 0]), abs(matrix[0, 1])
    out_w, out_h = int(w * cosine + h * sine), int(h * cosine + w * sine)
    matrix[0, 2] += (out_w - w) / 2
    matrix[1, 2] += (out_h - h) / 2
    return cv2.warpAffine(gray, matrix, (out_w, out_h), borderValue=255), float(angle)


def prepare_image(image: bytes) -> Image.Image:
    try:
        with Image.open(BytesIO(image)) as source:
            if source.width * source.height > 25_000_000:
                raise UnreadableReceipt("Image dimensions exceed limit")
            with ImageOps.exif_transpose(source) as upright:
                with upright.convert("L") as converted:
                    gray = np.array(converted)
        gray, rectified = rectify_receipt(gray)
        gray, angle = deskew(gray)
        scale = min(3.0 if max(gray.shape) < 1200 else 2.0, MAX_SIDE / max(gray.shape))
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        # Убираем тени без эрозии тонких штрихов.
        background = cv2.GaussianBlur(gray, (0, 0), 25)
        normalized = cv2.divide(gray, np.maximum(background, 1), scale=245)
        contrast = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)).apply(normalized)
        denoised = cv2.bilateralFilter(contrast, 3, 15, 15)
        sharpened = cv2.addWeighted(denoised, 1.25, cv2.GaussianBlur(denoised, (0, 0), 1), -.25, 0)
        prepared = Image.fromarray(sharpened)
        prepared.info["preprocessing"] = (
            f"exif,grayscale,crop_perspective={rectified},deskew={angle:.1f},"
            f"scale={scale:.2f},shadow_normalization,clahe,bilateral,light_sharpen"
        )
        return prepared
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError, cv2.error) as error:
        raise UnreadableReceipt("Cannot decode image") from error


def adaptive_image(image: Image.Image) -> Image.Image:
    # Grayscale остаётся отдельным кандидатом на случай потери тонких штрихов.
    binary = cv2.adaptiveThreshold(np.array(image), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 51, 13)
    return Image.fromarray(binary)
