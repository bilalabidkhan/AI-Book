"""
Object Detection Pipeline for Vision-Guided Manipulation

This example demonstrates a complete object detection pipeline for robotic manipulation,
including image preprocessing, detection, filtering, and result formatting.
"""

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from dataclasses import dataclass
from typing import List, Tuple, Optional
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


@dataclass
class DetectionResult:
    """
    Data class to represent a detection result
    """
    label: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    center: Tuple[float, float]  # (x, y)
    area: float


class ObjectDetectionPipeline:
    """
    Complete object detection pipeline for robotic manipulation
    """

    def __init__(self, confidence_threshold: float = 0.5, target_objects: Optional[List[str]] = None):
        """
        Initialize the detection pipeline

        Args:
            confidence_threshold: Minimum confidence for detections to be considered
            target_objects: List of specific objects to look for (None for all objects)
        """
        # Load pre-trained model with weights
        self.weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = fasterrcnn_resnet50_fpn(weights=self.weights)
        self.model.eval()

        # Set parameters
        self.confidence_threshold = confidence_threshold
        self.target_objects = target_objects or []  # Empty list means detect all objects

        # Get category names
        self.category_names = self.weights.meta["categories"]

        # Define transforms
        self.transform = T.Compose([
            T.ToTensor(),
        ])

        # Initialize CV bridge for ROS integration
        self.cv_bridge = CvBridge()

    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess the input image for detection

        Args:
            image: Input image as numpy array (H, W, C)

        Returns:
            Preprocessed image as tensor
        """
        # Ensure image is in RGB format (OpenCV loads as BGR)
        if image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image

        # Convert to PIL and then to tensor
        image_pil = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)  # PIL expects RGB
        image_tensor = self.transform(image_rgb)
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

        return image_tensor

    def detect_objects(self, image: np.ndarray) -> List[DetectionResult]:
        """
        Detect objects in the image using the pipeline

        Args:
            image: Input image as numpy array (H, W, C)

        Returns:
            List of DetectionResult objects
        """
        # Preprocess image
        image_tensor = self.preprocess_image(image)

        # Perform inference
        with torch.no_grad():
            predictions = self.model(image_tensor)

        # Extract results
        boxes = predictions[0]['boxes'].cpu()
        labels = predictions[0]['labels'].cpu()
        scores = predictions[0]['scores'].cpu()

        # Filter detections
        detections = []
        for i in range(len(boxes)):
            # Check confidence threshold
            if scores[i] < self.confidence_threshold:
                continue

            # Check if target object (if specific targets are specified)
            label_name = self.category_names[labels[i].item()]
            if self.target_objects and label_name not in self.target_objects:
                continue

            # Create DetectionResult
            bbox = boxes[i].tolist()
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

            detection = DetectionResult(
                label=label_name,
                confidence=scores[i].item(),
                bbox=tuple(bbox),
                center=(center_x, center_y),
                area=area
            )

            detections.append(detection)

        return detections

    def filter_detections_by_size(self, detections: List[DetectionResult],
                                min_area: float = 0.0, max_area: float = float('inf')) -> List[DetectionResult]:
        """
        Filter detections based on object size

        Args:
            detections: List of DetectionResult objects
            min_area: Minimum area threshold
            max_area: Maximum area threshold

        Returns:
            Filtered list of DetectionResult objects
        """
        return [det for det in detections if min_area <= det.area <= max_area]

    def filter_detections_by_location(self, detections: List[DetectionResult],
                                   image_shape: Tuple[int, int],
                                   x_range: Tuple[float, float] = (0.0, 1.0),
                                   y_range: Tuple[float, float] = (0.0, 1.0)) -> List[DetectionResult]:
        """
        Filter detections based on location in the image

        Args:
            detections: List of DetectionResult objects
            image_shape: (height, width) of the image
            x_range: Normalized x range (0.0 to 1.0)
            y_range: Normalized y range (0.0 to 1.0)

        Returns:
            Filtered list of DetectionResult objects
        """
        height, width = image_shape
        filtered_detections = []

        for detection in detections:
            center_x, center_y = detection.center
            norm_x = center_x / width
            norm_y = center_y / height

            if x_range[0] <= norm_x <= x_range[1] and y_range[0] <= norm_y <= y_range[1]:
                filtered_detections.append(detection)

        return filtered_detections

    def get_closest_object(self, detections: List[DetectionResult],
                          target_position: Tuple[float, float]) -> Optional[DetectionResult]:
        """
        Get the closest object to a target position

        Args:
            detections: List of DetectionResult objects
            target_position: Target (x, y) position

        Returns:
            Closest DetectionResult object or None if no detections
        """
        if not detections:
            return None

        target_x, target_y = target_position
        closest_detection = None
        min_distance = float('inf')

        for detection in detections:
            det_x, det_y = detection.center
            distance = np.sqrt((det_x - target_x)**2 + (det_y - target_y)**2)

            if distance < min_distance:
                min_distance = distance
                closest_detection = detection

        return closest_detection

    def process_ros_image(self, ros_image: Image) -> List[DetectionResult]:
        """
        Process a ROS Image message

        Args:
            ros_image: ROS Image message

        Returns:
            List of DetectionResult objects
        """
        # Convert ROS image to OpenCV format
        cv_image = self.cv_bridge.imgmsg_to_cv2(ros_image, desired_encoding='bgr8')
        return self.detect_objects(cv_image)


class ManipulationTargetSelector:
    """
    Class to select manipulation targets based on detection results
    """

    def __init__(self):
        self.preferred_objects = [
            'bottle', 'cup', 'book', 'apple', 'banana', 'orange', 'bowl',
            'fork', 'knife', 'spoon', 'sandwich', 'pizza', 'donut'
        ]

    def select_target(self, detections: List[DetectionResult]) -> Optional[DetectionResult]:
        """
        Select the best target for manipulation

        Args:
            detections: List of DetectionResult objects

        Returns:
            Best DetectionResult object for manipulation or None
        """
        if not detections:
            return None

        # First, try to find preferred objects
        preferred_detections = [
            det for det in detections
            if det.label in self.preferred_objects
        ]

        if preferred_detections:
            # Return the most confident preferred detection
            return max(preferred_detections, key=lambda x: x.confidence)

        # If no preferred objects, return the most confident detection
        return max(detections, key=lambda x: x.confidence)


# Example usage
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = ObjectDetectionPipeline(confidence_threshold=0.6)
    selector = ManipulationTargetSelector()

    # Create a dummy image for testing
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Detect objects
    detections = pipeline.detect_objects(dummy_image)

    print(f"Detected {len(detections)} objects:")
    for i, detection in enumerate(detections):
        print(f"  {i+1}. {detection.label} (confidence: {detection.confidence:.2f}, "
              f"center: {detection.center}, area: {detection.area:.2f})")

    # Filter by size (objects between 1000 and 50000 pixels)
    size_filtered = pipeline.filter_detections_by_size(detections, min_area=1000, max_area=50000)
    print(f"\nAfter size filtering: {len(size_filtered)} objects")

    # Select manipulation target
    target = selector.select_target(detections)
    if target:
        print(f"\nSelected manipulation target: {target.label} at {target.center}")
    else:
        print("\nNo suitable manipulation target found")