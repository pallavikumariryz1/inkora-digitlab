"""Image refinement: convert arbitrary canvas placement into a centred 28x28 image."""
from __future__ import annotations
import numpy as np
import cv2

def refine_digit(rgba: np.ndarray, output_size: int = 28) -> tuple[np.ndarray | None, dict]:
    """
    Convert a white/paper canvas with dark ink into MNIST-like white ink on black.
    The complete ink bounding box is preserved, padded, resized and centred by
    intensity-weighted centre of mass. Returns (image, diagnostics).
    """
    if rgba is None or not isinstance(rgba, np.ndarray) or rgba.size == 0:
        return None, {"reason": "No image was received."}
    arr = rgba
    if arr.ndim == 2:
        gray = arr.astype(np.uint8)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        rgb = arr[:, :, :3].astype(np.float32)
        alpha = arr[:, :, 3:4].astype(np.float32) / 255.0
        # Composite over white paper.
        rgb = rgb * alpha + 255.0 * (1.0 - alpha)
        gray = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    elif arr.ndim == 3 and arr.shape[2] == 3:
        gray = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        return None, {"reason": "Unsupported canvas image format."}

    # Dark pencil/ink on a light paper background -> bright foreground.
    ink = 255 - gray
    mask = (ink > 24).astype(np.uint8)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None, {"reason": "The canvas is blank."}

    h, w = gray.shape
    touches_edge = bool(xs.min() <= 1 or ys.min() <= 1 or xs.max() >= w - 2 or ys.max() >= h - 2)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    crop = ink[y0:y1, x0:x1]
    ch, cw = crop.shape

    # Preserve the entire drawing; add 20% margin around the longer dimension.
    side = max(ch, cw)
    margin = max(4, int(round(side * 0.20)))
    side += 2 * margin
    square = np.zeros((side, side), dtype=np.uint8)
    oy = (side - ch) // 2
    ox = (side - cw) // 2
    square[oy:oy + ch, ox:ox + cw] = crop

    # Fit into a 20x20 region within 28x28, retaining aspect ratio.
    target = 20
    scale = target / max(ch, cw)
    nw = max(1, int(round(cw * scale)))
    nh = max(1, int(round(ch * scale)))
    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    canvas = np.zeros((output_size, output_size), dtype=np.uint8)
    xoff, yoff = (output_size - nw) // 2, (output_size - nh) // 2
    canvas[yoff:yoff + nh, xoff:xoff + nw] = resized

    # Shift using intensity-weighted centre of mass (not canvas coordinates).
    moments = cv2.moments(canvas)
    if moments["m00"] > 0:
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
        dx = int(round((output_size - 1) / 2 - cx))
        dy = int(round((output_size - 1) / 2 - cy))
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        canvas = cv2.warpAffine(canvas, M, (output_size, output_size), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return canvas.astype(np.uint8), {
        "blank": False, "touches_edge": touches_edge, "bbox": (x0, y0, x1, y1),
        "original_size": (w, h), "centred": True, "ink_pixels": int(np.count_nonzero(mask))
    }
