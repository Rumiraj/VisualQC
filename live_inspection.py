import cv2
import time
import numpy as np
import torch
import torch.nn as nn

from torchvision import models, transforms
from PIL import Image

from paper_detection import preprocess_paper


# ---------------------------------
# CAMERA SETTINGS
# ---------------------------------

CAMERA_INDEX = 0

WIDTH = 1280
HEIGHT = 720
FPS = 60

PREVIEW_WIDTH = 960
PREVIEW_HEIGHT = 540

WINDOW_NAME = "VisualQC - AI Manufacturing Inspection"
FOCUS_WINDOW = "Camera Focus Controls"


# ---------------------------------
# INSPECTION SETTINGS
# ---------------------------------

EXPECTED_CIRCLES = 4

# Circle detection works on the 224x224 cropped paper.
# These limits are starting values and may need calibration.
MIN_AREA = 20
MAX_AREA = 3500

MIN_CIRCULARITY = 0.65

MODEL_PATH = "models/visualqc_resnet18_cropped.pth"


# ---------------------------------
# FOCUS SETTINGS
# ---------------------------------

autofocus_enabled = True

FOCUS_MAX = 250
FOCUS_DEFAULT = 100

focus_value = FOCUS_DEFAULT
focus_changed = False


# ---------------------------------
# GPU INITIALIZATION
# ---------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print()
print("================================")
print("       VISUALQC LIVE AI         ")
print("================================")

print(f"Device: {device}")

if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ---------------------------------
# LOAD TRAINED MODEL
# ---------------------------------

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

class_names = checkpoint["class_names"]

model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    len(class_names)
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)

model.eval()

print("Model loaded successfully!")
print(f"Classes: {class_names}")


# ---------------------------------
# IMAGE TRANSFORMATIONS
# ---------------------------------

# Must match eval_transform used during training.

transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# ---------------------------------
# OPENCV CIRCLE DETECTION
# ---------------------------------

def detect_circles(frame):

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    _, threshold = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(
        threshold,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )

    circle_count = 0
    detected_circles = []

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < MIN_AREA or area > MAX_AREA:
            continue

        perimeter = cv2.arcLength(
            contour,
            True
        )

        if perimeter == 0:
            continue

        circularity = (
            4 * np.pi * area
        ) / (perimeter ** 2)

        if circularity >= MIN_CIRCULARITY:

            circle_count += 1

            (x, y), radius = cv2.minEnclosingCircle(
                contour
            )

            detected_circles.append(
                (
                    int(x),
                    int(y),
                    int(radius)
                )
            )

    return circle_count, detected_circles


# ---------------------------------
# AI CLASSIFICATION
# ---------------------------------

def classify_image(frame):

    # Convert OpenCV BGR to RGB
    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    pil_image = Image.fromarray(rgb)

    input_tensor = transform(pil_image)

    input_tensor = input_tensor.unsqueeze(0)

    input_tensor = input_tensor.to(device)

    with torch.inference_mode():

        outputs = model(input_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, predicted = torch.max(
            probabilities,
            dim=1
        )

    predicted_class = class_names[
        predicted.item()
    ]

    confidence_percent = confidence.item() * 100

    return predicted_class, confidence_percent


# ---------------------------------
# CAMERA INITIALIZATION
# ---------------------------------

camera = cv2.VideoCapture(
    CAMERA_INDEX,
    cv2.CAP_DSHOW
)

if not camera.isOpened():
    raise RuntimeError(
        "ERROR: Camera could not be opened."
    )

# Enable MJPEG
camera.set(
    cv2.CAP_PROP_FOURCC,
    cv2.VideoWriter_fourcc(*"MJPG")
)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
camera.set(cv2.CAP_PROP_FPS, FPS)

camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Enable autofocus initially
camera.set(
    cv2.CAP_PROP_AUTOFOCUS,
    1
)

actual_width = int(
    camera.get(cv2.CAP_PROP_FRAME_WIDTH)
)

actual_height = int(
    camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

actual_fps = camera.get(
    cv2.CAP_PROP_FPS
)

print()
print("Camera initialized!")
print(f"Resolution: {actual_width}x{actual_height}")
print(f"Camera reported FPS: {actual_fps}")


# ---------------------------------
# FOCUS CONTROL
# ---------------------------------

def focus_slider(value):

    global focus_value
    global focus_changed

    focus_value = value
    focus_changed = True


def enable_autofocus():

    global autofocus_enabled

    success = camera.set(
        cv2.CAP_PROP_AUTOFOCUS,
        1
    )

    autofocus_enabled = True

    print(
        f"Autofocus enabled. "
        f"Driver accepted: {success}"
    )


def enable_manual_focus():

    global autofocus_enabled
    global focus_changed

    success = camera.set(
        cv2.CAP_PROP_AUTOFOCUS,
        0
    )

    autofocus_enabled = False
    focus_changed = True

    print(
        f"Manual focus enabled. "
        f"Driver accepted: {success}"
    )


def update_focus():

    global focus_changed

    if not autofocus_enabled and focus_changed:

        success = camera.set(
            cv2.CAP_PROP_FOCUS,
            focus_value
        )

        print(
            f"Focus: {focus_value} | "
            f"Driver accepted: {success}"
        )

        focus_changed = False


# ---------------------------------
# CREATE WINDOWS
# ---------------------------------

cv2.namedWindow(WINDOW_NAME)

cv2.namedWindow(FOCUS_WINDOW)

cv2.createTrackbar(
    "Focus",
    FOCUS_WINDOW,
    FOCUS_DEFAULT,
    FOCUS_MAX,
    focus_slider
)


# ---------------------------------
# FPS VARIABLES
# ---------------------------------

previous_time = time.perf_counter()
display_fps = 0.0


# ---------------------------------
# START LIVE INSPECTION
# ---------------------------------

print()
print("CONTROLS:")
print("A = Autofocus")
print("F = Manual focus")
print("Q = Quit")
print()
print("Inspection started.")


# ---------------------------------
# MAIN LOOP
# ---------------------------------

try:

    while True:

        # ---------------------------------
        # UPDATE FOCUS
        # ---------------------------------

        update_focus()

        # ---------------------------------
        # CAPTURE FRAME
        # ---------------------------------

        ret, frame = camera.read()

        if not ret:

            print("ERROR: Failed to capture frame.")
            break

        inspection_frame = cv2.resize(
            frame,
            (PREVIEW_WIDTH, PREVIEW_HEIGHT)
        )

        display = inspection_frame.copy()

        # ---------------------------------
        # PAPER DETECTION
        # ---------------------------------

        cropped_paper, paper_corners, paper_mask = preprocess_paper(
            inspection_frame
        )

        # Default state
        ai_result = "NO COMPONENT"
        opencv_result = "NO COMPONENT"

        confidence = 0.0
        inference_time = 0.0

        circle_count = 0
        circles = []

        # ---------------------------------
        # INSPECT PAPER IF DETECTED
        # ---------------------------------

        if cropped_paper is not None:

            # ---------------------------------
            # AI INFERENCE
            # ---------------------------------

            inference_start = time.perf_counter()

            predicted_class, confidence = classify_image(
                cropped_paper
            )

            if device.type == "cuda":
                torch.cuda.synchronize()

            inference_time = (
                time.perf_counter() - inference_start
            ) * 1000

            if predicted_class == "good":

                ai_result = "PASS"

            else:

                ai_result = "FAIL"

            # ---------------------------------
            # OPENCV INSPECTION
            # ---------------------------------

            circle_count, circles = detect_circles(
                cropped_paper
            )

            if circle_count == EXPECTED_CIRCLES:

                opencv_result = "PASS"

            elif circle_count < EXPECTED_CIRCLES:

                opencv_result = "FAIL"

            else:

                opencv_result = "INCONCLUSIVE"

            # ---------------------------------
            # DRAW PAPER BOUNDARIES
            # ---------------------------------

            corners_int = paper_corners.astype(
                np.int32
            )

            cv2.polylines(
                display,
                [corners_int],
                True,
                (255, 255, 0),
                2
            )

            # ---------------------------------
            # SHOW CROPPED PAPER
            # ---------------------------------

            cropped_display = cropped_paper.copy()

            for x, y, radius in circles:

                cv2.circle(
                    cropped_display,
                    (x, y),
                    radius,
                    (0, 255, 0),
                    2
                )

                cv2.circle(
                    cropped_display,
                    (x, y),
                    3,
                    (0, 0, 255),
                    -1
                )

            cv2.imshow(
                "Detected Paper",
                cropped_display
            )

        else:

            # ---------------------------------
            # NO PAPER DETECTED
            # ---------------------------------

            empty_preview = np.zeros(
                (224, 224, 3),
                dtype=np.uint8
            )

            cv2.putText(
                empty_preview,
                "NO COMPONENT",
                (20, 112),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

            cv2.imshow(
                "Detected Paper",
                empty_preview
            )

        # ---------------------------------
        # AI RESULT COLORS
        # ---------------------------------

        if ai_result == "PASS":

            ai_color = (0, 255, 0)

        elif ai_result == "FAIL":

            ai_color = (0, 0, 255)

        else:

            ai_color = (0, 255, 255)

        # ---------------------------------
        # FPS CALCULATION
        # ---------------------------------

        current_time = time.perf_counter()

        elapsed = current_time - previous_time

        previous_time = current_time

        if elapsed > 0:

            current_fps = 1.0 / elapsed

            display_fps = (
                0.9 * display_fps
                + 0.1 * current_fps
            )

        # ---------------------------------
        # DRAW INFORMATION PANEL
        # ---------------------------------

        overlay = display.copy()

        cv2.rectangle(
            overlay,
            (10, 10),
            (490, 310),
            (0, 0, 0),
            -1
        )

        cv2.addWeighted(
            overlay,
            0.65,
            display,
            0.35,
            0,
            display
        )

        # ---------------------------------
        # DISPLAY HEADER
        # ---------------------------------

        cv2.putText(
            display,
            "VISUALQC - LIVE INSPECTION",
            (25, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2
        )

        # ---------------------------------
        # DISPLAY AI RESULT
        # ---------------------------------

        cv2.putText(
            display,
            f"AI RESULT: {ai_result}",
            (25, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            ai_color,
            2
        )

        if cropped_paper is not None:

            confidence_text = (
                f"Confidence: {confidence:.1f}%"
            )

        else:

            confidence_text = "Confidence: N/A"

        cv2.putText(
            display,
            confidence_text,
            (25, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # ---------------------------------
        # DISPLAY OPENCV RESULT
        # ---------------------------------

        cv2.putText(
            display,
            f"OpenCV: {opencv_result}",
            (25, 165),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display,
            f"Circles: {circle_count}/{EXPECTED_CIRCLES}",
            (25, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # ---------------------------------
        # PERFORMANCE
        # ---------------------------------

        cv2.putText(
            display,
            f"AI inference: {inference_time:.1f} ms",
            (25, 245),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display,
            f"FPS: {display_fps:.1f}",
            (25, 280),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        # ---------------------------------
        # DISAGREEMENT DETECTION
        # ---------------------------------

        if (
            ai_result in ("PASS", "FAIL")
            and opencv_result in ("PASS", "FAIL")
            and ai_result != opencv_result
        ):

            cv2.putText(
                display,
                "WARNING: INSPECTION DISAGREEMENT",
                (200, 460),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

        # ---------------------------------
        # DISPLAY CAMERA FEED
        # ---------------------------------

        cv2.imshow(
            WINDOW_NAME,
            display
        )

        # ---------------------------------
        # KEYBOARD CONTROLS
        # ---------------------------------

        key = cv2.waitKey(1) & 0xFF

        # Enable autofocus
        if key == ord("a"):

            enable_autofocus()

        # Enable manual focus
        elif key == ord("f"):

            enable_manual_focus()

        # Quit
        elif key == ord("q"):

            break

finally:

    # ---------------------------------
    # CLEANUP
    # ---------------------------------

    camera.release()

    cv2.destroyAllWindows()

    print()
    print("VisualQC inspection stopped.")