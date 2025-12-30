"""
Object Detection Example for Vision-Guided Manipulation

This example demonstrates how to implement object detection for robotic manipulation
using PyTorch and torchvision models.
"""

import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from PIL import Image
import cv2
import numpy as np


class ObjectDetector:
    """
    Object detection class for robotic manipulation
    """

    def __init__(self, confidence_threshold=0.5):
        """
        Initialize the object detector
        """
        # Load pre-trained model with weights
        self.weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = fasterrcnn_resnet50_fpn(weights=self.weights)
        self.model.eval()

        # Set confidence threshold
        self.confidence_threshold = confidence_threshold

        # Get the category names
        self.category_names = self.weights.meta["categories"]

        # Define the transform
        self.transform = T.Compose([
            T.ToTensor(),
        ])

    def detect_objects(self, image_path):
        """
        Detect objects in an image

        Args:
            image_path: Path to the input image

        Returns:
            List of detected objects with bounding boxes and labels
        """
        # Load and preprocess the image
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image)
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

        # Perform inference
        with torch.no_grad():
            predictions = self.model(image_tensor)

        # Process predictions
        boxes = predictions[0]['boxes']
        labels = predictions[0]['labels']
        scores = predictions[0]['scores']

        # Filter detections based on confidence threshold
        valid_detections = scores > self.confidence_threshold

        detected_objects = []
        for i, valid in enumerate(valid_detections):
            if valid:
                obj = {
                    'bbox': boxes[i].tolist(),  # [x1, y1, x2, y2]
                    'label': self.category_names[labels[i].item()],
                    'confidence': scores[i].item()
                }
                detected_objects.append(obj)

        return detected_objects

    def detect_objects_from_cv2(self, cv2_image):
        """
        Detect objects from a cv2 image (numpy array)

        Args:
            cv2_image: OpenCV image (numpy array)

        Returns:
            List of detected objects with bounding boxes and labels
        """
        # Convert cv2 image (BGR) to PIL (RGB)
        image_rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)

        # Preprocess the image
        image_tensor = self.transform(image_pil)
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

        # Perform inference
        with torch.no_grad():
            predictions = self.model(image_tensor)

        # Process predictions
        boxes = predictions[0]['boxes']
        labels = predictions[0]['labels']
        scores = predictions[0]['scores']

        # Filter detections based on confidence threshold
        valid_detections = scores > self.confidence_threshold

        detected_objects = []
        for i, valid in enumerate(valid_detections):
            if valid:
                obj = {
                    'bbox': boxes[i].tolist(),  # [x1, y1, x2, y2]
                    'label': self.category_names[labels[i].item()],
                    'confidence': scores[i].item(),
                    'center': self._calculate_center(boxes[i].tolist())
                }
                detected_objects.append(obj)

        return detected_objects

    def _calculate_center(self, bbox):
        """
        Calculate the center point of a bounding box

        Args:
            bbox: [x1, y1, x2, y2]

        Returns:
            Center point (x, y)
        """
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        return (center_x, center_y)


def visualize_detections(image, detections, save_path=None):
    """
    Visualize object detections on an image

    Args:
        image: Input image (numpy array or cv2 image)
        detections: List of detected objects
        save_path: Path to save the output image (optional)

    Returns:
        Image with bounding boxes drawn
    """
    output_image = image.copy()

    for detection in detections:
        bbox = detection['bbox']
        label = detection['label']
        confidence = detection['confidence']

        # Draw bounding box
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw label and confidence
        label_text = f"{label}: {confidence:.2f}"
        cv2.putText(output_image, label_text, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save image if path provided
    if save_path:
        cv2.imwrite(save_path, output_image)

    return output_image


# Example usage
if __name__ == "__main__":
    # Initialize detector
    detector = ObjectDetector(confidence_threshold=0.6)

    # Example with a cv2 image (in a real scenario, this would come from a camera)
    # For this example, we'll create a dummy image
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Detect objects
    detections = detector.detect_objects_from_cv2(dummy_image)

    print(f"Detected {len(detections)} objects:")
    for i, obj in enumerate(detections):
        print(f"  {i+1}. {obj['label']} (confidence: {obj['confidence']:.2f}) at {obj['center']}")