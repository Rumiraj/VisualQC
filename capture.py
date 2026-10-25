import cv2
import os
import time
from datetime import datetime

# ---------------------------------
# CAMERA SETTINGS
# ---------------------------------

CAMERA_INDEX = 0

WIDTH = 1280
HEIGHT = 720
FPS = 60

PREVIEW_WIDTH = 960
PREVIEW_HEIGHT = 540

WINDOW_NAME = "Logitech C922 - Image Capture"

# ---------------------------------
# FOCUS SETTINGS
# ---------------------------------

focus_point = None
focus_size = 150
focus_requested = False

# ---------------------------------
# IMAGE DIRECTORY
# ---------------------------------

os.makedirs("images", exist_ok=True)

# ---------------------------------
# MOUSE CALLBACK
# ---------------------------------

def mouse_callback(event, x, y, flags, param):

    global focus_point
    global focus_requested

    if event == cv2.EVENT_LBUTTONDOWN:

        focus_point = (x, y)
        focus_requested = True

        print(f"Focus selected at: {x}, {y}")


# ---------------------------------
# CAMERA INITIALIZATION
# ---------------------------------

camera = cv2.VideoCapture(
    CAMERA_INDEX,
    cv2.CAP_DSHOW
)

if not camera.isOpened():
    print("ERROR: Camera could not be opened.")
    exit()

# Enable MJPEG
camera.set(
    cv2.CAP_PROP_FOURCC,
    cv2.VideoWriter_fourcc(*"MJPG")
)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
camera.set(cv2.CAP_PROP_FPS, FPS)

# Request a small capture buffer
camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Disable automatic focus
camera.set(cv2.CAP_PROP_AUTOFOCUS, 0)

# ---------------------------------
# CAMERA INFORMATION
# ---------------------------------

actual_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
actual_fps = camera.get(cv2.CAP_PROP_FPS)

print("Camera initialized successfully!")
print(f"Resolution: {actual_width}x{actual_height}")
print(f"Requested FPS: {FPS}")
print(f"Camera reported FPS: {actual_fps}")

print()
print("CONTROLS:")
print("Left Mouse Click = Focus")
print("S = Save Image")
print("Q = Quit")
print()

# ---------------------------------
# CREATE PREVIEW WINDOW
# ---------------------------------

cv2.namedWindow(WINDOW_NAME)

cv2.setMouseCallback(
    WINDOW_NAME,
    mouse_callback
)

# ---------------------------------
# AUTOFOCUS FUNCTION
# ---------------------------------

def perform_focus():

    global focus_point

    if focus_point is None:
        return

    print("Focusing...")

    # Convert preview coordinates to camera coordinates
    scale_x = actual_width / PREVIEW_WIDTH
    scale_y = actual_height / PREVIEW_HEIGHT

    x = int(focus_point[0] * scale_x)
    y = int(focus_point[1] * scale_y)

    half = focus_size // 2

    best_focus = None
    best_sharpness = -1

    # Try different lens focus positions
    for focus_value in range(0, 251, 10):

        success = camera.set(
            cv2.CAP_PROP_FOCUS,
            focus_value
        )

        if not success:
            print("Manual focus control is not supported.")
            return

        # Allow the lens to move
        time.sleep(0.12)

        # Discard potentially stale frames
        for _ in range(3):
            camera.grab()

        ret, frame = camera.read()

        if not ret:
            continue

        # Select region around clicked point
        x1 = max(0, x - half)
        y1 = max(0, y - half)

        x2 = min(actual_width, x + half)
        y2 = min(actual_height, y + half)

        region = frame[y1:y2, x1:x2]

        if region.size == 0:
            continue

        # Convert region to grayscale
        gray = cv2.cvtColor(
            region,
            cv2.COLOR_BGR2GRAY
        )

        # Calculate image sharpness
        sharpness = cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()

        if sharpness > best_sharpness:

            best_sharpness = sharpness
            best_focus = focus_value

    # Apply best focus position
    if best_focus is not None:

        camera.set(
            cv2.CAP_PROP_FOCUS,
            best_focus
        )

        print(f"Focus completed: {best_focus}")
        print(f"Sharpness score: {best_sharpness:.2f}")

    else:
        print("Focus adjustment failed.")


# ---------------------------------
# MAIN CAMERA LOOP
# ---------------------------------

previous_time = time.perf_counter()
display_fps = 0.0

try:

    while True:

        # ---------------------------------
        # CAPTURE FRAME
        # ---------------------------------

        ret, frame = camera.read()

        if not ret:
            print("ERROR: Failed to capture frame.")
            break

        # ---------------------------------
        # FPS CALCULATION
        # ---------------------------------

        current_time = time.perf_counter()

        elapsed = current_time - previous_time
        previous_time = current_time

        if elapsed > 0:

            current_fps = 1.0 / elapsed

            # Smooth FPS display
            display_fps = (
                0.9 * display_fps
                + 0.1 * current_fps
            )

        # ---------------------------------
        # RESIZE PREVIEW
        # ---------------------------------

        preview = cv2.resize(
            frame,
            (PREVIEW_WIDTH, PREVIEW_HEIGHT)
        )

        # ---------------------------------
        # DRAW FOCUS RECTANGLE
        # ---------------------------------

        if focus_point is not None:

            x, y = focus_point

            preview_half_x = int(
                focus_size * PREVIEW_WIDTH
                / actual_width / 2
            )

            preview_half_y = int(
                focus_size * PREVIEW_HEIGHT
                / actual_height / 2
            )

            cv2.rectangle(
                preview,
                (
                    x - preview_half_x,
                    y - preview_half_y
                ),
                (
                    x + preview_half_x,
                    y + preview_half_y
                ),
                (0, 255, 0),
                2
            )

            cv2.drawMarker(
                preview,
                (x, y),
                (0, 255, 0),
                cv2.MARKER_CROSS,
                20,
                2
            )

        # ---------------------------------
        # DISPLAY INFORMATION
        # ---------------------------------

        cv2.putText(
            preview,
            f"FPS: {display_fps:.1f}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            preview,
            "Click = Focus | S = Capture | Q = Quit",
            (20, PREVIEW_HEIGHT - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )

        # ---------------------------------
        # SHOW CAMERA
        # ---------------------------------

        cv2.imshow(
            WINDOW_NAME,
            preview
        )

        # ---------------------------------
        # KEYBOARD INPUT
        # ---------------------------------

        key = cv2.waitKey(1) & 0xFF

        # ---------------------------------
        # AUTOFOCUS
        # ---------------------------------

        if focus_requested:

            focus_requested = False

            perform_focus()

            # Reset FPS measurement after focus sweep
            previous_time = time.perf_counter()
            display_fps = 0.0

        # ---------------------------------
        # SAVE IMAGE
        # ---------------------------------

        if key == ord("s"):

            timestamp = datetime.now().strftime(
                "%Y%m%d_%H%M%S_%f"
            )

            filename = (
                f"images/capture_{timestamp}.jpg"
            )

            success = cv2.imwrite(
                filename,
                frame
            )

            if success:
                print(f"Image saved: {filename}")
            else:
                print("ERROR: Could not save image.")

        # ---------------------------------
        # EXIT PROGRAM
        # ---------------------------------

        elif key == ord("q"):
            break

finally:

    # ---------------------------------
    # CLEANUP
    # ---------------------------------

    camera.release()
    cv2.destroyAllWindows()

    print("Camera closed.")