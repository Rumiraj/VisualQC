# VisualQC — AI Based Visual Quality Inspection

**Python · OpenCV · PyTorch · ResNet18 · NVIDIA CUDA · Computer Vision**

VisualQC is a prototype manufacturing-quality inspection system developed during my Intelligent Systems Engineering studies. A Logitech C922 camera examines printed paper samples that represent manufactured parts with required circular features. The software first detects and straightens the paper using OpenCV, then combines two inspection methods: traditional contour-based circle counting and a ResNet18 neural-network classifier trained with PyTorch. The interface displays PASS, FAIL, or NO COMPONENT, with confidence scores, processing latency, and warnings when the two methods disagree.

The project covers data capture, manual image labeling, preprocessing, transfer learning, GPU-accelerated inference, and live camera integration. It also revealed engineering challenges such as class imbalance, high-confidence false acceptances, duplicate datasets, sensitivity to preprocessing, and the need for independent validation. I am extending the prototype with better dataset splits, robust component localization, and defect-specific evaluation metrics.

**My contribution:** End-to-end Python implementation, test-sample design, camera setup, dataset labeling, OpenCV inspection, PyTorch training, and real-time integration.

**Status:** Functional research/portfolio prototype; not production-qualified. Early-model holdout accuracy was 10/13 (76.92%) on a limited dataset and should not be interpreted as final system performance.

Suggested media: (1) camera view with four circles and agreement, (2) three-circle defect with detection overlay, (3) cropped component view and focus control, (4) comparison of model errors and planned improvements.
