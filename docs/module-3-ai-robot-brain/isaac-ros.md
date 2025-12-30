---
sidebar_position: 3
title: "Isaac ROS: Hardware-Accelerated Perception and VSLAM"
---

# Isaac ROS: Hardware-Accelerated Perception and VSLAM

This chapter covers hardware-accelerated perception using Isaac ROS, focusing on Visual Simultaneous Localization and Mapping (VSLAM) and real-time processing for AI-robot systems. Isaac ROS provides GPU-accelerated perception capabilities that enable real-time processing of sensor data.

## Introduction to Isaac ROS for Perception

Isaac ROS is a collection of high-performance perception packages designed to run on NVIDIA hardware. Key features include:

- **GPU acceleration**: Leverage NVIDIA GPUs for accelerated perception processing
- **Real-time performance**: Optimized for real-time applications with low latency
- **ROS/ROS2 integration**: Seamless integration with existing ROS/ROS2 workflows
- **Modular architecture**: Flexible, composable perception pipelines
- **Hardware optimization**: Specifically optimized for NVIDIA Jetson and discrete GPUs

Isaac ROS bridges the gap between high-performance perception algorithms and ROS-based robotics systems.

## Hardware Acceleration with NVIDIA GPUs

### GPU Resource Management
Isaac ROS leverages NVIDIA GPUs for accelerated processing:

```python
# Example GPU resource configuration
import rclpy
from rclpy.node import Node
import torch

class IsaacROSGPUManager(Node):
    def __init__(self):
        super().__init__('gpu_manager')

        # Check GPU availability
        self.gpu_available = torch.cuda.is_available()
        self.gpu_count = torch.cuda.device_count()

        if self.gpu_available:
            self.get_logger().info(f'Found {self.gpu_count} GPU(s)')
            for i in range(self.gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                self.get_logger().info(f'GPU {i}: {gpu_name}')
        else:
            self.get_logger().warn('No GPU found, falling back to CPU')
```

### CUDA Memory Management
Proper CUDA memory management is crucial for performance:

```cpp
// Example CUDA memory management in C++
#include <cuda_runtime.h>
#include <isaac_ros_common/gpu_memory_pool.hpp>

class PerceptionPipeline {
public:
    PerceptionPipeline() {
        // Initialize GPU memory pool
        gpu_memory_pool_ = std::make_unique<GPUMemoryPool>(1024 * 1024 * 256); // 256MB pool
    }

    void allocateGPUBuffer(size_t size) {
        gpu_buffer_ = gpu_memory_pool_->allocate(size);
    }

    void releaseGPUBuffer() {
        if (gpu_buffer_) {
            gpu_memory_pool_->release(gpu_buffer_);
            gpu_buffer_ = nullptr;
        }
    }

private:
    std::unique_ptr<GPUMemoryPool> gpu_memory_pool_;
    void* gpu_buffer_ = nullptr;
};
```

### TensorRT Integration
Isaac ROS integrates with TensorRT for optimized inference:

```python
# Example TensorRT integration
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

class TensorRTInference:
    def __init__(self, engine_path):
        self.engine = self.load_engine(engine_path)
        self.context = self.engine.create_execution_context()

    def load_engine(self, engine_path):
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        return runtime.deserialize_cuda_engine(engine_data)

    def infer(self, input_data):
        # Allocate I/O buffers
        inputs, outputs, bindings, stream = self.allocate_buffers()

        # Copy input data to GPU
        cuda.memcpy_htod(inputs[0].host, input_data)

        # Execute inference
        self.context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

        # Copy output data back to CPU
        cuda.memcpy_dtoh(outputs[0].host, outputs[0].device)

        return outputs[0].host
```

## Visual Simultaneous Localization and Mapping (VSLAM)

### VSLAM Architecture
Isaac ROS VSLAM system architecture:

```yaml
# VSLAM configuration example
vslam:
  tracking:
    feature_detector:
      type: "orb"
      max_features: 2000
      scale_factor: 1.2
      levels: 8
      edge_threshold: 31
      patch_size: 31

    feature_matcher:
      max_distance: 75
      max_ratio: 0.8

    pose_estimator:
      ransac_threshold: 5.0
      min_inliers: 10

  mapping:
    local_map_size: 10.0
    keyframe_threshold: 0.1
    bundle_adjustment:
      enabled: true
      max_iterations: 100

  loop_closure:
    detection:
      enabled: true
      distance_threshold: 0.5
    correction:
      enabled: true
      max_iterations: 50
```

### VSLAM Node Implementation
Example VSLAM node implementation:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from visualization_msgs.msg import MarkerArray
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

class IsaacROSVisualSLAM(Node):
    def __init__(self):
        super().__init__('isaac_ros_vslam')

        # Subscriptions
        self.image_sub = self.create_subscription(
            Image,
            '/camera/rgb/image_raw',
            self.image_callback,
            10
        )

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/rgb/camera_info',
            self.camera_info_callback,
            10
        )

        # Publishers
        self.odom_pub = self.create_publisher(Odometry, '/vslam/odometry', 10)
        self.pose_pub = self.create_publisher(PoseStamped, '/vslam/pose', 10)
        self.map_pub = self.create_publisher(MarkerArray, '/vslam/map', 10)

        # VSLAM state
        self.camera_matrix = None
        self.distortion_coeffs = None
        self.previous_frame = None
        self.current_pose = np.eye(4)
        self.keyframes = []
        self.map_points = []

        # Feature detection
        self.feature_detector = cv2.ORB_create(nfeatures=2000)
        self.feature_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def camera_info_callback(self, msg):
        self.camera_matrix = np.array(msg.k).reshape(3, 3)
        self.distortion_coeffs = np.array(msg.d)

    def image_callback(self, msg):
        # Convert ROS image to OpenCV
        image = self.ros_image_to_cv2(msg)

        if self.previous_frame is None:
            # Initialize first frame
            self.initialize_frame(image)
        else:
            # Process frame for tracking
            pose_update = self.track_frame(self.previous_frame, image)
            if pose_update is not None:
                self.update_pose(pose_update)
                self.publish_odometry()

        self.previous_frame = image

    def initialize_frame(self, image):
        # Extract features from initial frame
        kp = self.feature_detector.detect(image)
        kp, des = self.feature_detector.compute(image, kp)

        self.initial_features = (kp, des)

    def track_frame(self, prev_frame, curr_frame):
        # Extract features
        kp_prev, des_prev = self.initial_features
        kp_curr, des_curr = self.feature_detector.detectAndCompute(curr_frame, None)

        # Match features
        matches = self.feature_matcher.knnMatch(des_prev, des_curr, k=2)

        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.8 * n.distance:
                    good_matches.append(m)

        if len(good_matches) >= 10:
            # Extract matched points
            src_pts = np.float32([kp_prev[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_curr[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # Estimate pose using Essential Matrix
            E, mask = cv2.findEssentialMat(src_pts, dst_pts, self.camera_matrix,
                                         method=cv2.RANSAC, threshold=5.0)

            if E is not None:
                # Recover pose
                _, R, t, _ = cv2.recoverPose(E, src_pts, dst_pts, self.camera_matrix)

                # Create transformation matrix
                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = t.flatten()

                return T

        return None

    def update_pose(self, pose_update):
        # Update current pose
        self.current_pose = np.dot(self.current_pose, pose_update)

    def publish_odometry(self):
        odom_msg = Odometry()
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.frame_id = 'map'
        odom_msg.child_frame_id = 'base_link'

        # Set position
        odom_msg.pose.pose.position.x = self.current_pose[0, 3]
        odom_msg.pose.pose.position.y = self.current_pose[1, 3]
        odom_msg.pose.pose.position.z = self.current_pose[2, 3]

        # Convert rotation matrix to quaternion
        r = R.from_matrix(self.current_pose[:3, :3])
        quat = r.as_quat()
        odom_msg.pose.pose.orientation.x = quat[0]
        odom_msg.pose.pose.orientation.y = quat[1]
        odom_msg.pose.pose.orientation.z = quat[2]
        odom_msg.pose.pose.orientation.w = quat[3]

        self.odom_pub.publish(odom_msg)

    def ros_image_to_cv2(self, ros_image):
        # Convert ROS Image message to OpenCV image
        dtype = np.uint8
        if ros_image.encoding == 'rgb8':
            dtype = np.uint8
        elif ros_image.encoding == 'rgba8':
            dtype = np.uint8
        elif ros_image.encoding == 'bgr8':
            dtype = np.uint8
        elif ros_image.encoding == 'mono8':
            dtype = np.uint8
        elif ros_image.encoding == 'mono16':
            dtype = np.uint16

        img = np.frombuffer(ros_image.data, dtype=dtype).reshape(
            ros_image.height, ros_image.width, -1
        )

        if ros_image.encoding == 'rgb8':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        return img
```

## Sensor Data Processing Pipelines

### Pipeline Architecture
Isaac ROS sensor processing pipeline architecture:

```yaml
# Isaac ROS sensor processing pipeline configuration
pipeline:
  input_topics:
    - /camera/rgb/image_raw
    - /camera/depth/image_raw
    - /imu/data
    - /lidar/scan

  processing_nodes:
    - image_processing:
        node_type: "isaac_ros_image_proc"
        parameters:
          image_width: 640
          image_height: 480
          rectify: true

    - stereo_vision:
        node_type: "isaac_ros_stereo_image_proc"
        parameters:
          baseline: 0.1
          focal_length: 320.0

    - point_cloud:
        node_type: "isaac_ros_point_cloud_proc"
        parameters:
          input_type: "depth"
          output_frame: "base_link"

    - object_detection:
        node_type: "isaac_ros_detection"
        parameters:
          model_path: "/models/yolo_v5.pt"
          confidence_threshold: 0.5
          nms_threshold: 0.4
```

### Example Pipeline Implementation
Complete pipeline example:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, Imu
from geometry_msgs.msg import Twist
from std_msgs.msg import Header
from cv_bridge import CvBridge
import cv2
import numpy as np
from threading import Lock

class IsaacROSPipeline(Node):
    def __init__(self):
        super().__init__('isaac_ros_pipeline')

        # Initialize components
        self.cv_bridge = CvBridge()
        self.pipeline_lock = Lock()

        # Subscriptions
        self.image_sub = self.create_subscription(
            Image, '/camera/rgb/image_raw', self.image_callback, 10
        )

        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10
        )

        # Publishers
        self.perception_pub = self.create_publisher(
            PointCloud2, '/perception/pointcloud', 10
        )

        self.command_pub = self.create_publisher(
            Twist, '/perception/cmd_vel', 10
        )

        # Processing components
        self.image_processor = ImageProcessor()
        self.perception_pipeline = PerceptionPipeline()

        # Processing timer
        self.processing_timer = self.create_timer(
            0.033, self.process_pipeline  # ~30 FPS
        )

        self.latest_image = None
        self.latest_imu = None

    def image_callback(self, msg):
        with self.pipeline_lock:
            self.latest_image = self.cv_bridge.imgmsg_to_cv2(msg, 'bgr8')

    def imu_callback(self, msg):
        with self.pipeline_lock:
            self.latest_imu = msg

    def process_pipeline(self):
        with self.pipeline_lock:
            if self.latest_image is not None:
                # Process image through perception pipeline
                processed_data = self.image_processor.process(self.latest_image)

                # Run perception algorithms
                perception_result = self.perception_pipeline.run(processed_data)

                # Publish results
                if perception_result is not None:
                    self.publish_perception_results(perception_result)

    def publish_perception_results(self, results):
        # Publish processed point cloud
        pc_msg = self.create_pointcloud_msg(results)
        self.perception_pub.publish(pc_msg)

        # Publish control commands based on perception
        cmd_msg = self.generate_control_command(results)
        self.command_pub.publish(cmd_msg)

class ImageProcessor:
    def __init__(self):
        self.feature_detector = cv2.ORB_create(nfeatures=1000)

    def process(self, image):
        # Preprocess image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Extract features
        kp, des = self.feature_detector.detectAndCompute(gray, None)

        # Return processed data
        return {
            'image': image,
            'gray': gray,
            'features': (kp, des),
            'timestamp': cv2.getTickCount()
        }

class PerceptionPipeline:
    def __init__(self):
        # Initialize perception models
        self.object_detector = self.load_object_detector()
        self.depth_estimator = self.load_depth_estimator()

    def load_object_detector(self):
        # Load object detection model
        # This would typically load a TensorRT engine
        return None

    def load_depth_estimator(self):
        # Load depth estimation model
        return None

    def run(self, processed_data):
        # Run perception pipeline
        image = processed_data['image']
        gray = processed_data['gray']

        # Object detection
        objects = self.detect_objects(image)

        # Depth estimation
        depth_map = self.estimate_depth(image)

        # Return perception results
        return {
            'objects': objects,
            'depth': depth_map,
            'features': processed_data['features'],
            'timestamp': processed_data['timestamp']
        }

    def detect_objects(self, image):
        # Run object detection
        # This would typically use a trained model
        return []

    def estimate_depth(self, image):
        # Estimate depth from image
        # This would typically use a depth estimation model
        return np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
```

## Real-time Performance Optimization

### Performance Monitoring
Monitor real-time performance:

```python
import time
from collections import deque

class PerformanceMonitor:
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.processing_times = deque(maxlen=window_size)
        self.fps_history = deque(maxlen=window_size)
        self.last_time = time.time()

    def start_timing(self):
        self.start_time = time.time()

    def end_timing(self):
        end_time = time.time()
        processing_time = end_time - self.start_time
        self.processing_times.append(processing_time)

        # Calculate FPS
        current_time = time.time()
        elapsed = current_time - self.last_time
        if elapsed > 0:
            fps = 1.0 / elapsed
            self.fps_history.append(fps)
        self.last_time = current_time

        return processing_time, self.get_average_fps()

    def get_average_fps(self):
        if len(self.fps_history) > 0:
            return sum(self.fps_history) / len(self.fps_history)
        return 0.0

    def get_average_processing_time(self):
        if len(self.processing_times) > 0:
            return sum(self.processing_times) / len(self.processing_times)
        return 0.0
```

### Threading and Asynchronous Processing
Optimize with threading:

```python
import threading
import queue
from concurrent.futures import ThreadPoolExecutor

class AsyncPerceptionPipeline:
    def __init__(self, num_threads=4):
        self.executor = ThreadPoolExecutor(max_workers=num_threads)
        self.input_queue = queue.Queue(maxsize=10)
        self.output_queue = queue.Queue(maxsize=10)

        # Start processing thread
        self.processing_thread = threading.Thread(target=self.process_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()

    def submit_image(self, image):
        try:
            self.input_queue.put_nowait(image)
        except queue.Full:
            # Drop frame if queue is full
            pass

    def get_results(self):
        try:
            return self.output_queue.get_nowait()
        except queue.Empty:
            return None

    def process_loop(self):
        while True:
            try:
                image = self.input_queue.get(timeout=1.0)

                # Submit to thread pool for processing
                future = self.executor.submit(self.process_image, image)

                # Add result to output queue
                result = future.result()
                try:
                    self.output_queue.put_nowait(result)
                except queue.Full:
                    pass  # Drop result if output queue is full
            except queue.Empty:
                continue  # Continue loop

    def process_image(self, image):
        # Process image with Isaac ROS components
        # This would include feature extraction, object detection, etc.
        return self.run_perception_algorithms(image)

    def run_perception_algorithms(self, image):
        # Placeholder for perception algorithms
        return {'processed': True, 'timestamp': time.time()}
```

## Integration with Existing ROS Systems

### ROS 2 Integration
Isaac ROS integration with ROS 2 ecosystem:

```python
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Twist
from std_msgs.msg import String

class IsaacROSIntegration(Node):
    def __init__(self):
        super().__init__('isaac_ros_integration')

        # Define QoS profiles for different data types
        image_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )

        cmd_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE
        )

        # Create subscribers with appropriate QoS
        self.image_sub = self.create_subscription(
            Image, '/camera/rgb/image_raw',
            self.image_callback, image_qos
        )

        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/camera/rgb/camera_info',
            self.camera_info_callback, image_qos
        )

        # Create publishers
        self.perception_pub = self.create_publisher(
            String, '/isaac_ros/perception_results', cmd_qos
        )

        self.cmd_pub = self.create_publisher(
            Twist, '/cmd_vel', cmd_qos
        )

    def image_callback(self, msg):
        # Process image through Isaac ROS pipeline
        self.process_image_with_isaac_ros(msg)

    def process_image_with_isaac_ros(self, image_msg):
        # Integrate with Isaac ROS perception nodes
        # This would typically involve calling Isaac ROS services
        # or publishing to Isaac ROS-specific topics
        pass
```

### Parameter Configuration
Example parameter configuration for Isaac ROS:

```yaml
# Isaac ROS parameter configuration
isaac_ros_perception:
  ros__parameters:
    # Performance settings
    processing_frequency: 30.0
    max_queue_size: 10

    # Hardware acceleration
    use_gpu: true
    gpu_device_id: 0
    cuda_memory_pool_size: 268435456  # 256 MB

    # Feature detection
    feature_detector:
      max_features: 2000
      scale_factor: 1.2
      levels: 8
      edge_threshold: 31
      patch_size: 31

    # Tracking parameters
    tracking:
      max_feature_age: 100
      min_feature_distance: 10
      tracking_threshold: 0.9

    # Memory management
    memory:
      enable_pooling: true
      pool_size: 104857600  # 100 MB
      max_allocation_size: 10485760  # 10 MB
```

## Perception Pipeline Configurations and VSLAM Setup Examples

### Complete VSLAM Configuration
Full example of a VSLAM setup:

```python
#!/usr/bin/env python3
"""
Complete Isaac ROS VSLAM example
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, Imu
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from visualization_msgs.msg import MarkerArray
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from cv_bridge import CvBridge

class CompleteVSLAMNode(Node):
    def __init__(self):
        super().__init__('complete_vslam')

        # Initialize components
        self.cv_bridge = CvBridge()
        self.feature_detector = cv2.ORB_create(nfeatures=2000)
        self.feature_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # Camera parameters
        self.camera_matrix = None
        self.distortion_coeffs = None

        # VSLAM state
        self.previous_frame = None
        self.current_pose = np.eye(4)
        self.keyframes = []
        self.map_points = []
        self.local_map_size = 10.0  # meters

        # Subscriptions
        self.image_sub = self.create_subscription(
            Image, '/camera/rgb/image_raw', self.image_callback, 10
        )

        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/camera/rgb/camera_info', self.camera_info_callback, 10
        )

        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10
        )

        # Publishers
        self.odom_pub = self.create_publisher(Odometry, '/vslam/odometry', 10)
        self.pose_pub = self.create_publisher(PoseStamped, '/vslam/pose', 10)
        self.map_pub = self.create_publisher(MarkerArray, '/vslam/map', 10)

        # Performance monitoring
        self.frame_count = 0
        self.start_time = self.get_clock().now()

        self.get_logger().info('Complete VSLAM node initialized')

    def camera_info_callback(self, msg):
        self.camera_matrix = np.array(msg.k).reshape(3, 3)
        self.distortion_coeffs = np.array(msg.d)
        self.get_logger().info('Camera info received')

    def imu_callback(self, msg):
        # Use IMU data for better pose estimation
        # This is a simplified example
        pass

    def image_callback(self, msg):
        if self.camera_matrix is None:
            return  # Wait for camera info

        # Convert ROS image to OpenCV
        image = self.cv_bridge.imgmsg_to_cv2(msg, 'bgr8')

        if self.previous_frame is None:
            # Initialize first frame
            self.initialize_frame(image, msg.header.stamp)
        else:
            # Process frame for tracking
            pose_update = self.track_frame(self.previous_frame, image)
            if pose_update is not None:
                self.update_pose(pose_update, msg.header.stamp)
                self.publish_odometry(msg.header.stamp)

        self.previous_frame = image
        self.frame_count += 1

        # Log performance every 100 frames
        if self.frame_count % 100 == 0:
            current_time = self.get_clock().now()
            elapsed = (current_time - self.start_time).nanoseconds / 1e9
            fps = self.frame_count / elapsed
            self.get_logger().info(f'Processed {self.frame_count} frames, FPS: {fps:.2f}')

    def initialize_frame(self, image, stamp):
        # Extract features from initial frame
        kp = self.feature_detector.detect(image)
        kp, des = self.feature_detector.compute(image, kp)

        self.initial_features = (kp, des)
        self.get_logger().info(f'Initialized frame with {len(kp)} features')

    def track_frame(self, prev_frame, curr_frame):
        # Extract features
        kp_prev, des_prev = self.initial_features
        kp_curr, des_curr = self.feature_detector.detectAndCompute(curr_frame, None)

        if des_prev is None or des_curr is None or len(des_prev) < 10 or len(des_curr) < 10:
            return None

        # Match features
        matches = self.feature_matcher.knnMatch(des_prev, des_curr, k=2)

        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.8 * n.distance:
                    good_matches.append(m)

        if len(good_matches) >= 10:
            # Extract matched points
            src_pts = np.float32([kp_prev[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_curr[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # Estimate pose using Essential Matrix
            E, mask = cv2.findEssentialMat(src_pts, dst_pts, self.camera_matrix,
                                         method=cv2.RANSAC, threshold=5.0)

            if E is not None:
                # Recover pose
                _, R_mat, t, _ = cv2.recoverPose(E, src_pts, dst_pts, self.camera_matrix)

                # Create transformation matrix
                T = np.eye(4)
                T[:3, :3] = R_mat
                T[:3, 3] = t.flatten()

                # Update initial features for next frame
                self.initial_features = (kp_curr, des_curr)

                return T

        return None

    def update_pose(self, pose_update, stamp):
        # Update current pose
        self.current_pose = np.dot(self.current_pose, pose_update)

        # Add keyframe if significant movement
        if self.should_add_keyframe():
            self.add_keyframe(stamp)

    def should_add_keyframe(self):
        # Add keyframe if movement is significant
        translation = np.linalg.norm(self.current_pose[:3, 3])
        if len(self.keyframes) == 0 or translation > 0.5:  # 50cm threshold
            return True
        return False

    def add_keyframe(self, stamp):
        # Add current pose as keyframe
        self.keyframes.append((self.current_pose.copy(), stamp))

        # Limit keyframes to local map size
        while len(self.keyframes) > 100:  # Keep last 100 keyframes
            self.keyframes.pop(0)

    def publish_odometry(self, stamp):
        odom_msg = Odometry()
        odom_msg.header.stamp = stamp
        odom_msg.header.frame_id = 'map'
        odom_msg.child_frame_id = 'base_link'

        # Set position
        odom_msg.pose.pose.position.x = self.current_pose[0, 3]
        odom_msg.pose.pose.position.y = self.current_pose[1, 3]
        odom_msg.pose.pose.position.z = self.current_pose[2, 3]

        # Convert rotation matrix to quaternion
        r = R.from_matrix(self.current_pose[:3, :3])
        quat = r.as_quat()
        odom_msg.pose.pose.orientation.x = quat[0]
        odom_msg.pose.pose.orientation.y = quat[1]
        odom_msg.pose.pose.orientation.z = quat[2]
        odom_msg.pose.pose.orientation.w = quat[3]

        # Set velocities (approximate)
        if len(self.keyframes) > 1:
            prev_pose = self.keyframes[-2][0]
            dt = 0.033  # Assume 30 FPS
            vel = (self.current_pose[:3, 3] - prev_pose[:3, 3]) / dt
            odom_msg.twist.twist.linear.x = vel[0]
            odom_msg.twist.twist.linear.y = vel[1]
            odom_msg.twist.twist.linear.z = vel[2]

        self.odom_pub.publish(odom_msg)

        # Publish pose
        pose_msg = PoseStamped()
        pose_msg.header = odom_msg.header
        pose_msg.pose = odom_msg.pose.pose
        self.pose_pub.publish(pose_msg)

def main(args=None):
    rclpy.init(args=args)
    node = CompleteVSLAMNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Prerequisites

To effectively work with Isaac ROS for hardware-accelerated perception, you should have:
- Understanding of ROS/ROS2 concepts and message types
- Basic knowledge of computer vision and feature detection
- Familiarity with NVIDIA GPU computing (CUDA, TensorRT)
- Experience with Python or C++ programming
- Understanding of sensor data processing (optional but helpful)

This chapter provides the foundation for implementing hardware-accelerated perception systems using Isaac ROS. The next chapter will cover Nav2 for humanoid navigation with path planning and motion control.