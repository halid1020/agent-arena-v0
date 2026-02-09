import cv2
import numpy as np
import math

MILD_RED = (120, 120, 220)  # soft red (B, G, R)

TEXT_Y_STEP = 35
TEXT_Y_STATUS = 80
TEXT_BG_ALPHA = 0.6

BLUE = (200, 50, 50)   # left picker
RED  = (50, 50, 200)   # right picker
MILD_RED = (120, 120, 220)  # soft red (B, G, R)

PRIMITIVE_COLORS = {
    "norm-pixel-pick-and-place": (255, 180, 80),          # orange
    "norm-pixel-pick-and-fling": (80, 200, 255),# cyan
    "no-operation": (255, 255, 255),             # gray
    "default": (255, 255, 255),                  # white
}

# -------------------------------
# Normalized → pixel coordinates
# -------------------------------
def norm_to_px(v, W, H):
    x = int((v[0] + 1) * 0.5 * W)
    y = int((v[1] + 1) * 0.5 * H)
    return x, y

def apply_workspace_shade(rgb, mask, color, alpha=0.35):
    """
    Shade pixels where mask == True with given BGR color.
    """
    shaded = rgb.copy()
    overlay = np.zeros_like(rgb, dtype=np.uint8)
    overlay[:] = color

    shaded[mask] = cv2.addWeighted(
        rgb[mask], 1 - alpha,
        overlay[mask], alpha,
        0
    )
    return shaded


# OpenCV uses (row, col)
def swap(p):
    return (p[1], p[0])

def draw_big_arrowhead(img, p_from, p_to, color, size=28):
    """
    p_from, p_to are (x, y)
    """
    dx = p_to[0] - p_from[0]
    dy = p_to[1] - p_from[1]
    norm = np.sqrt(dx * dx + dy * dy) + 1e-6

    ux, uy = dx / norm, dy / norm      # direction
    px, py = -uy, ux                   # perpendicular

    tip = np.array([p_to[0], p_to[1]])

    left = np.array([
        p_to[0] - size * ux + size * 0.6 * px,
        p_to[1] - size * uy + size * 0.6 * py
    ])

    right = np.array([
        p_to[0] - size * ux - size * 0.6 * px,
        p_to[1] - size * uy - size * 0.6 * py
    ])

    # Convert to OpenCV (col, row) only here
    pts = np.array([
        (int(tip[0]),   int(tip[1])),
        (int(left[0]),  int(left[1])),
        (int(right[0]), int(right[1]))
    ], dtype=np.int32)

    cv2.fillPoly(img, [pts], color)

def draw_colored_line(img, p_start, p_end, cmap, thickness=8, num_samples=20):
    """
    Draw a straight line from p_start → p_end with color gradient.
    p_start, p_end: (x, y) in pixel coords
    """
    xs = np.linspace(p_start[0], p_end[0], num_samples).astype(int)
    ys = np.linspace(p_start[1], p_end[1], num_samples).astype(int)

    for i in range(1, num_samples):
        alpha = i / (num_samples - 1)

        value = np.uint8([[[int((1.0 - alpha) * 255)]]])
        color = cv2.applyColorMap(value, cmap)[0, 0].tolist()

        cv2.line(
            img,
            (xs[i - 1], ys[i - 1]),
            (xs[i], ys[i]),
            color,
            thickness
        )

    draw_big_arrowhead(
        img,
        (xs[-2], ys[-2]),
        (xs[-1], ys[-1]),
        color,
        size=30
    )

def draw_text_with_bg(img, text, org, color, scale=1.2, thickness=4):
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = org
    overlay = img.copy()
    cv2.rectangle(
        overlay,
        (x - 5, y - h - 5),
        (x + w + 5, y + 5),
        (0, 0, 0),
        -1
    )
    cv2.addWeighted(overlay, TEXT_BG_ALPHA, img, 1 - TEXT_BG_ALPHA, 0, img)
    cv2.putText(
        img,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA
    )

def draw_pick_and_place(img, action):
    H, W = img.shape[:2]
    if len(action) > 4:
        pick_0  = norm_to_px(action[:2], W, H)
        pick_1  = norm_to_px(action[2:4], W, H)
        place_0 = norm_to_px(action[4:6], W, H)
        place_1 = norm_to_px(action[6:8], W, H)

        # ----- Ensure BLUE pick is always the LEFT one -----
        picks = [(pick_0, place_0), (pick_1, place_1)]
        picks_sorted = sorted(picks, key=lambda p: p[0][1])  # sort by x

        (left_pick, left_place), (right_pick, right_place) = picks_sorted
        
    else:

        left_pick  = norm_to_px(action[:2], W, H)
        left_place  = norm_to_px(action[2:4], W, H)

   
    # BLUE = left, RED = right

    # ----- Draw arrows with SMALLER arrowheads -----
    small_tip = 0.08    # << smaller arrowhead tip size

    # Colormaps (same convention as fling)
    cmap_left  = cv2.COLORMAP_COOL   # BLUE-ish
    cmap_right = cv2.COLORMAP_AUTUMN   # RED-ish

    draw_colored_line(
        img,
        swap(left_pick),
        swap(left_place),
        cmap_left,
        thickness=8,
        num_samples=20
    )

    if len(action) > 4:
        draw_colored_line(
            img,
            swap(right_pick),
            swap(right_place),
            cmap_right,
            thickness=8,
            num_samples=20
        )


    BLUE = cv2.applyColorMap(
        np.uint8([[[0]]]), cmap_left
    )[0, 0].tolist()
                    
    RED = cv2.applyColorMap(
        np.uint8([[[0]]]), cmap_right
    )[0, 0].tolist()
    
    # ----- Hollow circles -----
    cv2.circle(img, swap(left_pick),  10, BLUE, 3)

    if len(action) > 4:
        cv2.circle(img, swap(right_pick), 10, RED,  3)

    return img

def draw_pick_and_place_with_orient(img, pick_px, place_px, pick_orient=None, place_orient=None):
    H, W = img.shape[:2]
    
    # --- Helper: Draw Clean Gradient Orientation Line ---
    def draw_gradient_orientation(img, center_px, theta, cmap, length=50, is_ref=False):
        # We use a custom swap for the reference if it's already in (x, y)
        if is_ref:
            cx, cy = center_px
        else:
            cx, cy = swap(center_px) 
        
        adjusted_theta = theta + (math.pi / 2)
        dx = math.cos(adjusted_theta)
        dy = -math.sin(adjusted_theta) 
        
        p_start = (int(cx - (length/2) * dx), int(cy - (length/2) * dy))
        p_end   = (int(cx + (length/2) * dx), int(cy + (length/2) * dy))
        
        num_samples = 15
        xs = np.linspace(p_start[0], p_end[0], num_samples).astype(int)
        ys = np.linspace(p_start[1], p_end[1], num_samples).astype(int)

        for i in range(1, num_samples):
            alpha = i / (num_samples - 1)
            val = int(alpha * 255)
            color = cv2.applyColorMap(np.uint8([[[val]]]), cmap)[0, 0].tolist()
            cv2.line(img, (xs[i-1], ys[i-1]), (xs[i], ys[i]), color, 3, cv2.LINE_AA)
            cv2.line(img, (xs[i-1], ys[i-1]), (xs[i], ys[i]), color, 3, cv2.LINE_AA)

    # ---------------------------------------------------------
    # Main Drawing Logic
    # ---------------------------------------------------------
    
    # Trajectory uses COOL (Cyan/Magenta)
    cmap_move = cv2.COLORMAP_COOL 
    # Orientation uses BONE (Black/White/Gray) or SPRING (Magenta/Yellow)
    # Let's use BONE for a high-contrast 'mechanical' feel
    cmap_orient = cv2.COLORMAP_BONE

    # 1. Draw the movement trajectory arrow
    draw_colored_line(
        img,
        swap(pick_px),
        swap(place_px),
        cmap_move,
        thickness=6,
        num_samples=20
    )

    if pick_orient is not None:
        draw_gradient_orientation(img, pick_px, pick_orient, cmap_orient)
    if place_orient is not None:
        draw_gradient_orientation(img, place_px, place_orient, cmap_orient)

    # --- NEW: Reference Orientation at Bottom Right ---
    ref_length = 40
    # Position: 40 pixels from right, 40 pixels from bottom
    ref_center = (W - 50, H - 50) 
    
    # Draw the 0-degree line
    draw_gradient_orientation(img, ref_center, 0, cmap_orient, length=ref_length, is_ref=True)
    
    # Add "0°" text label next to it
    cv2.putText(
        img, "0", (W - 85, H - 42), 
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
    )

    return img