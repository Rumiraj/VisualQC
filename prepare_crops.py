import os
import cv2

from paper_detection import preprocess_paper

# ---------------------------------
# SETTINGS
# ---------------------------------

SOURCE_FOLDER = "data"
OUTPUT_FOLDER = "data_cropped"

CATEGORIES = [
    "good",
    "defective"
]

# ---------------------------------
# PROCESS DATASET
# ---------------------------------

total_processed = 0
total_successful = 0
total_failed = 0

for category in CATEGORIES:

    source = os.path.join(
        SOURCE_FOLDER,
        category
    )

    destination = os.path.join(
        OUTPUT_FOLDER,
        category
    )

    os.makedirs(
        destination,
        exist_ok=True
    )

    images = sorted([
        filename
        for filename in os.listdir(source)
        if filename.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ])

    print()
    print(f"Processing category: {category}")

    for filename in images:

        total_processed += 1

        filepath = os.path.join(
            source,
            filename
        )

        image = cv2.imread(filepath)

        if image is None:

            total_failed += 1

            print(f"FAILED TO READ: {filename}")
            continue

        # Detect and crop paper
        cropped, corners, mask = preprocess_paper(
            image
        )

        if cropped is None:

            total_failed += 1

            print(f"PAPER NOT FOUND: {filename}")
            continue

        output_path = os.path.join(
            destination,
            os.path.splitext(filename)[0] + ".png"
        )

        # Do not overwrite existing cropped images
        if os.path.exists(output_path):

            print(f"ALREADY EXISTS: {output_path}")
            continue

        success = cv2.imwrite(
            output_path,
            cropped
        )

        if success:

            total_successful += 1

            print(f"SAVED: {output_path}")

        else:

            total_failed += 1

            print(f"SAVE FAILED: {filename}")

# ---------------------------------
# SUMMARY
# ---------------------------------

print()
print("================================")
print("       CROPPING COMPLETE        ")
print("================================")

print(f"Images processed: {total_processed}")
print(f"New crops saved: {total_successful}")
print(f"Detection/save failures: {total_failed}")

print("================================")