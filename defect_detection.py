import cv2
import os
import shutil
import numpy as np

# ---------------------------------
# SETTINGS
# ---------------------------------

IMAGE_FOLDER = "images"
DATA_FOLDER = "data"

EXPECTED_CIRCLES = 4

MIN_AREA = 200
MAX_AREA = 20000

MIN_CIRCULARITY = 0.65

IMAGE_WIDTH = 960
IMAGE_HEIGHT = 540

# ---------------------------------
# DATASET DIRECTORIES
# ---------------------------------

GOOD_FOLDER = os.path.join(DATA_FOLDER, "good")
DEFECTIVE_FOLDER = os.path.join(DATA_FOLDER, "defective")
INCONCLUSIVE_FOLDER = os.path.join(DATA_FOLDER, "inconclusive")

os.makedirs(GOOD_FOLDER, exist_ok=True)
os.makedirs(DEFECTIVE_FOLDER, exist_ok=True)
os.makedirs(INCONCLUSIVE_FOLDER, exist_ok=True)

# ---------------------------------
# INSPECTION COUNTERS
# ---------------------------------

pass_count = 0
fail_count = 0
inconclusive_count = 0
total_inspected = 0

# ---------------------------------
# MANUAL CLASSIFICATION COUNTERS
# ---------------------------------

good_saved = 0
defective_saved = 0
inconclusive_saved = 0
skipped_count = 0

# ---------------------------------
# GENERATE NEXT AVAILABLE FILENAME
# ---------------------------------

def get_next_filename(folder, prefix, extension=".jpg"):

    number = 1

    while True:

        filename = f"{prefix}_{number:03d}{extension}"

        filepath = os.path.join(
            folder,
            filename
        )

        if not os.path.exists(filepath):
            return filepath

        number += 1


# ---------------------------------
# SAVE CLASSIFIED IMAGE
# ---------------------------------

def save_classified_image(source, category):

    if category == "good":

        folder = GOOD_FOLDER
        prefix = "good"

    elif category == "defective":

        folder = DEFECTIVE_FOLDER
        prefix = "defective"

    elif category == "inconclusive":

        folder = INCONCLUSIVE_FOLDER
        prefix = "inconclusive"

    else:
        return False

    extension = os.path.splitext(source)[1].lower()

    destination = get_next_filename(
        folder,
        prefix,
        extension
    )

    # Prevent accidental overwriting
    if os.path.exists(destination):
        print(f"ERROR: File already exists: {destination}")
        return False

    try:

        shutil.copy2(
            source,
            destination
        )

        print(f"Saved: {destination}")
        return True

    except OSError as error:

        print(f"ERROR: Could not save image: {error}")
        return False


# ---------------------------------
# LOAD IMAGES
# ---------------------------------

if not os.path.isdir(IMAGE_FOLDER):

    print(f"ERROR: Folder not found: {IMAGE_FOLDER}")
    exit()

images = sorted([
    filename
    for filename in os.listdir(IMAGE_FOLDER)
    if filename.lower().endswith((".jpg", ".jpeg", ".png"))
])

if not images:

    print("ERROR: No images found!")
    exit()

print()
print("================================")
print("     VISUALQC INSPECTION        ")
print("================================")

print(f"Images found: {len(images)}")

print()
print("CONTROLS:")
print("G = Good")
print("D = Defective")
print("I = Inconclusive")
print("S = Skip")
print("Q = Quit")

print("================================")
print()

# ---------------------------------
# PROCESS IMAGES
# ---------------------------------

for filename in images:

    filepath = os.path.join(
        IMAGE_FOLDER,
        filename
    )

    image = cv2.imread(filepath)

    if image is None:

        print(f"Could not load: {filename}")
        continue

    # Resize image for processing
    image = cv2.resize(
        image,
        (IMAGE_WIDTH, IMAGE_HEIGHT)
    )

    # ---------------------------------
    # GRAYSCALE
    # ---------------------------------

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # ---------------------------------
    # REMOVE NOISE
    # ---------------------------------

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # ---------------------------------
    # THRESHOLD IMAGE
    # ---------------------------------

    _, threshold = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # ---------------------------------
    # FIND CONTOURS
    # ---------------------------------

    contours, _ = cv2.findContours(
        threshold,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )

    circle_count = 0

    # ---------------------------------
    # DETECT CIRCLES
    # ---------------------------------

    for contour in contours:

        area = cv2.contourArea(contour)

        # Ignore shapes outside area limits
        if area < MIN_AREA or area > MAX_AREA:
            continue

        perimeter = cv2.arcLength(
            contour,
            True
        )

        if perimeter == 0:
            continue

        # Calculate circularity
        circularity = (
            4 * np.pi * area
        ) / (perimeter ** 2)

        # ---------------------------------
        # VALID CIRCLE
        # ---------------------------------

        if circularity >= MIN_CIRCULARITY:

            circle_count += 1

            (x, y), radius = cv2.minEnclosingCircle(
                contour
            )

            center = (
                int(x),
                int(y)
            )

            radius = int(radius)

            # Draw detected circle
            cv2.circle(
                image,
                center,
                radius,
                (0, 255, 0),
                2
            )

            # Mark circle center
            cv2.circle(
                image,
                center,
                3,
                (0, 0, 255),
                -1
            )

    # ---------------------------------
    # AUTOMATIC QUALITY INSPECTION
    # ---------------------------------

    if circle_count == EXPECTED_CIRCLES:

        result = "PASS"
        color = (0, 255, 0)

        pass_count += 1

    elif circle_count < EXPECTED_CIRCLES:

        result = "FAIL"
        color = (0, 0, 255)

        fail_count += 1

    else:

        result = "INCONCLUSIVE"
        color = (0, 255, 255)

        inconclusive_count += 1

    total_inspected += 1

    # ---------------------------------
    # DISPLAY INSPECTION RESULT
    # ---------------------------------

    cv2.putText(
        image,
        f"STATUS: {result}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )

    cv2.putText(
        image,
        f"Circles detected: {circle_count}/{EXPECTED_CIRCLES}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2
    )

    # ---------------------------------
    # DISPLAY INSPECTION COUNTERS
    # ---------------------------------

    cv2.putText(
        image,
        f"Inspected: {total_inspected}",
        (20, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        image,
        f"PASS: {pass_count}",
        (20, 165),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        image,
        f"FAIL: {fail_count}",
        (20, 200),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

    cv2.putText(
        image,
        f"INCONCLUSIVE: {inconclusive_count}",
        (20, 235),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    # ---------------------------------
    # DISPLAY MANUAL CLASSIFICATION
    # ---------------------------------

    cv2.putText(
        image,
        f"Dataset GOOD: {good_saved}",
        (20, 285),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2
    )

    cv2.putText(
        image,
        f"Dataset DEFECTIVE: {defective_saved}",
        (20, 320),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 255),
        2
    )

    cv2.putText(
        image,
        f"Dataset INCONCLUSIVE: {inconclusive_saved}",
        (20, 355),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2
    )

    # ---------------------------------
    # KEYBOARD INSTRUCTIONS
    # ---------------------------------

    cv2.putText(
        image,
        "G = Good | D = Defective | I = Inconclusive",
        (20, 470),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.putText(
        image,
        "S = Skip | Q = Quit",
        (20, 505),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    # ---------------------------------
    # TERMINAL OUTPUT
    # ---------------------------------

    print()
    print("--------------------------------")

    print(f"Image: {filename}")
    print(f"Circles detected: {circle_count}")
    print(f"Automatic result: {result}")

    print("--------------------------------")

    # ---------------------------------
    # SHOW IMAGES
    # ---------------------------------

    cv2.imshow(
        "Manufacturing Inspection",
        image
    )

    cv2.imshow(
        "Threshold Image",
        threshold
    )

    # ---------------------------------
    # MANUAL CLASSIFICATION
    # ---------------------------------

    while True:

        key = cv2.waitKey(0) & 0xFF

        # GOOD
        if key == ord("g"):

            if save_classified_image(filepath, "good"):

                good_saved += 1
                break

        # DEFECTIVE
        elif key == ord("d"):

            if save_classified_image(filepath, "defective"):

                defective_saved += 1
                break

        # INCONCLUSIVE
        elif key == ord("i"):

            if save_classified_image(filepath, "inconclusive"):

                inconclusive_saved += 1
                break

        # SKIP
        elif key == ord("s"):

            skipped_count += 1

            print(f"Skipped: {filename}")
            break

        # QUIT
        elif key == ord("q"):

            print("Inspection stopped by user.")

            cv2.destroyAllWindows()

            print()
            print("================================")
            print("       INSPECTION SUMMARY       ")
            print("================================")

            print(f"Total inspected: {total_inspected}")
            print(f"PASS: {pass_count}")
            print(f"FAIL: {fail_count}")
            print(f"INCONCLUSIVE: {inconclusive_count}")

            print()
            print("MANUAL DATASET CLASSIFICATION")
            print(f"GOOD: {good_saved}")
            print(f"DEFECTIVE: {defective_saved}")
            print(f"INCONCLUSIVE: {inconclusive_saved}")
            print(f"SKIPPED: {skipped_count}")

            print("================================")

            exit()

# ---------------------------------
# CLEANUP
# ---------------------------------

cv2.destroyAllWindows()

# ---------------------------------
# FINAL SUMMARY
# ---------------------------------

print()
print("================================")
print("       INSPECTION SUMMARY       ")
print("================================")

print(f"Total inspected: {total_inspected}")

print()
print("AUTOMATIC INSPECTION")
print(f"PASS:         {pass_count}")
print(f"FAIL:         {fail_count}")
print(f"INCONCLUSIVE: {inconclusive_count}")

print()
print("MANUAL DATASET CLASSIFICATION")
print(f"GOOD:         {good_saved}")
print(f"DEFECTIVE:    {defective_saved}")
print(f"INCONCLUSIVE: {inconclusive_saved}")
print(f"SKIPPED:      {skipped_count}")

print("================================")