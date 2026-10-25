import cv2
import numpy as np

# ---------------------------------
# SETTINGS
# ---------------------------------

OUTPUT_SIZE = 224

MIN_PAPER_AREA = 0.05
MAX_PAPER_AREA = 0.90

MIN_RECTANGULARITY = 0.70

# Minimum brightness of the paper
PAPER_THRESHOLD = 110


# ---------------------------------
# ORDER CORNER POINTS
# ---------------------------------

def order_points(points):

    points = np.array(
        points,
        dtype=np.float32
    ).reshape(4, 2)

    # Sort corners by vertical position
    y_sorted = points[np.argsort(points[:, 1])]

    top = y_sorted[:2]
    bottom = y_sorted[2:]

    top = top[np.argsort(top[:, 0])]
    bottom = bottom[np.argsort(bottom[:, 0])]

    top_left = top[0]
    top_right = top[1]

    bottom_left = bottom[0]
    bottom_right = bottom[1]

    return np.array([
        top_left,
        top_right,
        bottom_right,
        bottom_left
    ], dtype=np.float32)


# ---------------------------------
# DETECT PAPER
# ---------------------------------

def detect_paper(frame):

    height, width = frame.shape[:2]

    frame_area = height * width

    # Convert to grayscale
    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # Reduce noise
    blurred = cv2.GaussianBlur(
        gray,
        (7, 7),
        0
    )

    # Separate bright paper from dark background
    _, mask = cv2.threshold(
        blurred,
        PAPER_THRESHOLD,
        255,
        cv2.THRESH_BINARY
    )

    # Fill small gaps in detected paper
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (9, 9)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    # Find outer contours
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Largest contours first
    contours = sorted(
        contours,
        key=cv2.contourArea,
        reverse=True
    )

    for contour in contours:

        area = cv2.contourArea(contour)

        area_ratio = area / frame_area

        # Reject objects that are too small or large
        if not (
            MIN_PAPER_AREA
            <= area_ratio
            <= MAX_PAPER_AREA
        ):
            continue

        # Fit a rotated rectangle
        rectangle = cv2.minAreaRect(contour)

        box = cv2.boxPoints(rectangle)

        rect_width, rect_height = rectangle[1]

        if rect_width <= 0 or rect_height <= 0:
            continue

        rectangle_area = rect_width * rect_height

        # Check how closely the contour fills the rectangle
        rectangularity = area / rectangle_area

        if rectangularity < MIN_RECTANGULARITY:
            continue

        # Reject extremely narrow shapes
        aspect_ratio = max(
            rect_width,
            rect_height
        ) / min(
            rect_width,
            rect_height
        )

        if aspect_ratio > 2.0:
            continue

        # Paper detected
        corners = order_points(box)

        return corners, mask

    return None, mask


# ---------------------------------
# CROP AND STRAIGHTEN PAPER
# ---------------------------------

def crop_paper(frame, corners):

    destination = np.array([
        [0, 0],
        [OUTPUT_SIZE - 1, 0],
        [OUTPUT_SIZE - 1, OUTPUT_SIZE - 1],
        [0, OUTPUT_SIZE - 1]
    ], dtype=np.float32)

    # Calculate perspective transformation
    matrix = cv2.getPerspectiveTransform(
        corners,
        destination
    )

    # Warp the paper to a square image
    cropped = cv2.warpPerspective(
        frame,
        matrix,
        (OUTPUT_SIZE, OUTPUT_SIZE)
    )

    return cropped


# ---------------------------------
# COMPLETE PREPROCESSING
# ---------------------------------

def preprocess_paper(frame):

    corners, mask = detect_paper(frame)

    if corners is None:
        return None, None, mask

    cropped = crop_paper(
        frame,
        corners
    )

    return cropped, corners, mask