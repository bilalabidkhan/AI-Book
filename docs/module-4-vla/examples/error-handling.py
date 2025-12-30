"""
Error Handling Across VLA Components

This example demonstrates comprehensive error handling strategies for Vision-Language-Action systems.
"""

import openai
import cv2
import numpy as np
import torch
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import logging
from enum import Enum


class ErrorType(Enum):
    """
    Enum for different types of errors in VLA systems
    """
    VISION_ERROR = "vision_error"
    LANGUAGE_ERROR = "language_error"
    ACTION_ERROR = "action_error"
    COMMUNICATION_ERROR = "communication_error"
    SAFETY_ERROR = "safety_error"
    TIMEOUT_ERROR = "timeout_error"
    VALIDATION_ERROR = "validation_error"


@dataclass
class ErrorInfo:
    """
    Data class for error information
    """
    error_type: ErrorType
    message: str
    component: str
    timestamp: float
    severity: str  # "low", "medium", "high", "critical"
    recovery_strategy: str
    context: Dict[str, Any]


class VLARetryHandler:
    """
    Handler for retry logic in VLA systems
    """

    def __init__(self, max_retries: int = 3, base_delay: float = 1.0):
        """
        Initialize retry handler

        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Base delay between retries (seconds)
        """
        self.max_retries = max_retries
        self.base_delay = base_delay

    def execute_with_retry(self, func, *args, **kwargs) -> Tuple[bool, Any, Optional[ErrorInfo]]:
        """
        Execute a function with retry logic

        Args:
            func: Function to execute
            *args: Arguments to pass to function
            **kwargs: Keyword arguments to pass to function

        Returns:
            Tuple of (success, result, error_info)
        """
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                result = func(*args, **kwargs)
                return True, result, None
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    # Exponential backoff
                    delay = self.base_delay * (2 ** attempt)
                    print(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    print(f"All {self.max_retries} retry attempts failed.")
                    error_info = ErrorInfo(
                        error_type=ErrorType.COMMUNICATION_ERROR,
                        message=str(e),
                        component=func.__name__,
                        timestamp=time.time(),
                        severity="high",
                        recovery_strategy="Manual intervention required",
                        context={"function": func.__name__, "args": args, "kwargs": kwargs}
                    )
                    return False, None, error_info

        return False, None, last_error


class VisionErrorHandler:
    """
    Error handling for vision components
    """

    def __init__(self):
        self.recovery_strategies = {
            ErrorType.VISION_ERROR: [
                "Adjust camera parameters",
                "Change lighting conditions",
                "Use alternative detection model",
                "Request user to reposition robot"
            ],
            ErrorType.TIMEOUT_ERROR: [
                "Increase timeout threshold",
                "Use lower resolution processing",
                "Switch to faster detection algorithm"
            ]
        }

    def handle_detection_error(self, image: np.ndarray, error: Exception) -> ErrorInfo:
        """
        Handle errors in object detection

        Args:
            image: Input image that caused the error
            error: Exception that occurred

        Returns:
            ErrorInfo object
        """
        error_msg = str(error)
        error_type = ErrorType.VISION_ERROR

        if "timeout" in error_msg.lower():
            error_type = ErrorType.TIMEOUT_ERROR
        elif "cuda" in error_msg.lower() or "memory" in error_msg.lower():
            error_type = ErrorType.VISION_ERROR

        recovery_strategy = self._select_recovery_strategy(error_type)

        error_info = ErrorInfo(
            error_type=error_type,
            message=error_msg,
            component="Vision System",
            timestamp=time.time(),
            severity=self._determine_severity(error_type),
            recovery_strategy=recovery_strategy,
            context={
                "image_shape": image.shape if image is not None else None,
                "image_dtype": str(image.dtype) if image is not None else None
            }
        )

        return error_info

    def _select_recovery_strategy(self, error_type: ErrorType) -> str:
        """
        Select appropriate recovery strategy based on error type

        Args:
            error_type: Type of error

        Returns:
            Recovery strategy
        """
        strategies = self.recovery_strategies.get(error_type, ["Fallback to default behavior"])
        return strategies[0] if strategies else "Manual intervention required"

    def _determine_severity(self, error_type: ErrorType) -> str:
        """
        Determine severity level based on error type

        Args:
            error_type: Type of error

        Returns:
            Severity level
        """
        severity_map = {
            ErrorType.SAFETY_ERROR: "critical",
            ErrorType.TIMEOUT_ERROR: "high",
            ErrorType.VISION_ERROR: "medium",
            ErrorType.VALIDATION_ERROR: "low"
        }
        return severity_map.get(error_type, "medium")


class LanguageErrorHandler:
    """
    Error handling for language components
    """

    def __init__(self):
        self.recovery_strategies = {
            ErrorType.LANGUAGE_ERROR: [
                "Use alternative parsing method",
                "Request clarification from user",
                "Fallback to default action",
                "Switch to simpler command interpretation"
            ],
            ErrorType.COMMUNICATION_ERROR: [
                "Use cached model response",
                "Switch to local language model",
                "Request user to repeat command"
            ]
        }

    def handle_language_error(self, command: str, error: Exception) -> ErrorInfo:
        """
        Handle errors in language processing

        Args:
            command: Command that caused the error
            error: Exception that occurred

        Returns:
            ErrorInfo object
        """
        error_msg = str(error)
        error_type = ErrorType.LANGUAGE_ERROR

        if "api" in error_msg.lower() or "connection" in error_msg.lower():
            error_type = ErrorType.COMMUNICATION_ERROR
        elif "timeout" in error_msg.lower():
            error_type = ErrorType.TIMEOUT_ERROR

        recovery_strategy = self._select_recovery_strategy(error_type)

        error_info = ErrorInfo(
            error_type=error_type,
            message=error_msg,
            component="Language System",
            timestamp=time.time(),
            severity=self._determine_severity(error_type),
            recovery_strategy=recovery_strategy,
            context={
                "command_length": len(command),
                "command_preview": command[:50] + "..." if len(command) > 50 else command
            }
        )

        return error_info

    def _select_recovery_strategy(self, error_type: ErrorType) -> str:
        """
        Select appropriate recovery strategy based on error type
        """
        strategies = self.recovery_strategies.get(error_type, ["Fallback to default behavior"])
        return strategies[0] if strategies else "Manual intervention required"

    def _determine_severity(self, error_type: ErrorType) -> str:
        """
        Determine severity level based on error type
        """
        severity_map = {
            ErrorType.SAFETY_ERROR: "critical",
            ErrorType.COMMUNICATION_ERROR: "high",
            ErrorType.LANGUAGE_ERROR: "medium",
            ErrorType.VALIDATION_ERROR: "low"
        }
        return severity_map.get(error_type, "medium")


class ActionErrorHandler:
    """
    Error handling for action components
    """

    def __init__(self):
        self.recovery_strategies = {
            ErrorType.ACTION_ERROR: [
                "Abort current action",
                "Return to safe position",
                "Retry with modified parameters",
                "Use alternative manipulation strategy"
            ],
            ErrorType.SAFETY_ERROR: [
                "Emergency stop",
                "Return to home position",
                "Alert human operator",
                "Disable robot movement"
            ]
        }

    def handle_action_error(self, action: Dict[str, Any], error: Exception) -> ErrorInfo:
        """
        Handle errors in action execution

        Args:
            action: Action that caused the error
            error: Exception that occurred

        Returns:
            ErrorInfo object
        """
        error_msg = str(error)
        error_type = ErrorType.ACTION_ERROR

        if "collision" in error_msg.lower() or "safety" in error_msg.lower():
            error_type = ErrorType.SAFETY_ERROR
        elif "timeout" in error_msg.lower():
            error_type = ErrorType.TIMEOUT_ERROR

        recovery_strategy = self._select_recovery_strategy(error_type)

        error_info = ErrorInfo(
            error_type=error_type,
            message=error_msg,
            component="Action System",
            timestamp=time.time(),
            severity=self._determine_severity(error_type),
            recovery_strategy=recovery_strategy,
            context={
                "action_type": action.get('action_type', 'unknown'),
                "action_description": action.get('description', ''),
                "action_parameters": action.get('parameters', {})
            }
        )

        return error_info

    def _select_recovery_strategy(self, error_type: ErrorType) -> str:
        """
        Select appropriate recovery strategy based on error type
        """
        strategies = self.recovery_strategies.get(error_type, ["Fallback to default behavior"])
        return strategies[0] if strategies else "Manual intervention required"

    def _determine_severity(self, error_type: ErrorType) -> str:
        """
        Determine severity level based on error type
        """
        severity_map = {
            ErrorType.SAFETY_ERROR: "critical",
            ErrorType.ACTION_ERROR: "high",
            ErrorType.TIMEOUT_ERROR: "medium",
            ErrorType.VALIDATION_ERROR: "low"
        }
        return severity_map.get(error_type, "medium")


class VLAErrorManager:
    """
    Centralized error management for the entire VLA system
    """

    def __init__(self):
        self.vision_handler = VisionErrorHandler()
        self.language_handler = LanguageErrorHandler()
        self.action_handler = ActionErrorHandler()
        self.retry_handler = VLARetryHandler()
        self.error_log: List[ErrorInfo] = []

    def log_error(self, error_info: ErrorInfo):
        """
        Log an error to the error log

        Args:
            error_info: ErrorInfo object to log
        """
        self.error_log.append(error_info)
        print(f"[ERROR] {error_info.component}: {error_info.message}")
        print(f"  Type: {error_info.error_type.value}, Severity: {error_info.severity}")
        print(f"  Recovery: {error_info.recovery_strategy}")

    def handle_vision_error(self, image: np.ndarray, error: Exception) -> ErrorInfo:
        """
        Handle vision component error and log it

        Args:
            image: Input image that caused the error
            error: Exception that occurred

        Returns:
            ErrorInfo object
        """
        error_info = self.vision_handler.handle_detection_error(image, error)
        self.log_error(error_info)
        return error_info

    def handle_language_error(self, command: str, error: Exception) -> ErrorInfo:
        """
        Handle language component error and log it

        Args:
            command: Command that caused the error
            error: Exception that occurred

        Returns:
            ErrorInfo object
        """
        error_info = self.language_handler.handle_language_error(command, error)
        self.log_error(error_info)
        return error_info

    def handle_action_error(self, action: Dict[str, Any], error: Exception) -> ErrorInfo:
        """
        Handle action component error and log it

        Args:
            action: Action that caused the error
            error: Exception that occurred

        Returns:
            ErrorInfo object
        """
        error_info = self.action_handler.handle_action_error(action, error)
        self.log_error(error_info)
        return error_info

    def get_error_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about logged errors

        Returns:
            Dictionary with error statistics
        """
        if not self.error_log:
            return {"total_errors": 0}

        total_errors = len(self.error_log)
        error_types = {}
        severities = {}
        components = {}

        for error in self.error_log:
            # Count error types
            error_type = error.error_type.value
            error_types[error_type] = error_types.get(error_type, 0) + 1

            # Count severities
            severity = error.severity
            severities[severity] = severities.get(severity, 0) + 1

            # Count components
            component = error.component
            components[component] = components.get(component, 0) + 1

        return {
            "total_errors": total_errors,
            "error_types": error_types,
            "severities": severities,
            "components": components,
            "most_common_error": max(error_types, key=error_types.get) if error_types else None
        }

    def execute_with_error_handling(self, func, component_name: str, *args, **kwargs) -> Tuple[bool, Any]:
        """
        Execute a function with comprehensive error handling

        Args:
            func: Function to execute
            component_name: Name of the component
            *args: Arguments to pass to function
            **kwargs: Keyword arguments to pass to function

        Returns:
            Tuple of (success, result)
        """
        try:
            # Try execution with retry logic
            success, result, error_info = self.retry_handler.execute_with_retry(func, *args, **kwargs)

            if success:
                return True, result
            else:
                # Log the error that occurred after all retries failed
                if error_info:
                    self.log_error(error_info)
                return False, None

        except Exception as e:
            # Handle unexpected exceptions
            error_msg = str(e)
            error_info = ErrorInfo(
                error_type=ErrorType.COMMUNICATION_ERROR,
                message=error_msg,
                component=component_name,
                timestamp=time.time(),
                severity="critical",
                recovery_strategy="Manual intervention required",
                context={"function": func.__name__, "args": args, "kwargs": kwargs}
            )
            self.log_error(error_info)
            return False, None


class SafeVLASystem:
    """
    A VLA system with built-in safety and error handling
    """

    def __init__(self):
        self.error_manager = VLAErrorManager()
        self.safety_thresholds = {
            "collision_distance": 0.1,  # meters
            "execution_timeout": 30.0,  # seconds
            "vision_timeout": 5.0,      # seconds
            "language_timeout": 10.0    # seconds
        }

    def safe_vision_processing(self, image: np.ndarray) -> Optional[List[Dict[str, Any]]]:
        """
        Safely process vision input with error handling

        Args:
            image: Input image

        Returns:
            List of detected objects or None if error
        """
        def _vision_func(img):
            # Simulate vision processing (in real implementation, this would call actual vision functions)
            if img is None or img.size == 0:
                raise ValueError("Invalid image input")
            # Simulate processing time
            time.sleep(0.1)
            # Return dummy objects
            return [
                {"id": "obj1", "label": "cup", "confidence": 0.85, "bbox": [100, 100, 150, 150]},
                {"id": "obj2", "label": "book", "confidence": 0.78, "bbox": [200, 200, 250, 250]}
            ]

        success, result = self.error_manager.execute_with_error_handling(
            _vision_func, "Vision System", image
        )

        if success:
            return result
        else:
            print("Vision processing failed, returning empty object list")
            return []

    def safe_language_processing(self, command: str) -> Optional[List[Dict[str, Any]]]:
        """
        Safely process language input with error handling

        Args:
            command: Natural language command

        Returns:
            List of planned actions or None if error
        """
        def _language_func(cmd):
            # Simulate language processing
            if not cmd or len(cmd.strip()) == 0:
                raise ValueError("Empty command")
            # Simulate processing time
            time.sleep(0.05)
            # Return dummy action plan
            return [
                {"action_type": "navigation", "description": f"Process command: {cmd}", "parameters": {}}
            ]

        success, result = self.error_manager.execute_with_error_handling(
            _language_func, "Language System", command
        )

        if success:
            return result
        else:
            print("Language processing failed, returning default action")
            return [{"action_type": "wait", "description": "Error recovery - waiting for new command", "parameters": {}}]

    def safe_action_execution(self, actions: List[Dict[str, Any]]) -> bool:
        """
        Safely execute actions with error handling

        Args:
            actions: List of actions to execute

        Returns:
            True if all actions executed successfully, False otherwise
        """
        def _action_func(action_list):
            # Simulate action execution
            for i, action in enumerate(action_list):
                # Simulate execution time
                time.sleep(0.1)
                # Check for simulated failures
                if np.random.random() < 0.1:  # 10% chance of failure
                    raise RuntimeError(f"Action {i} failed during execution")
            return True

        success, result = self.error_manager.execute_with_error_handling(
            _action_func, "Action System", actions
        )

        return success if success else False

    def process_command_with_safety(self, command: str, image: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Process a command with full safety and error handling

        Args:
            command: Natural language command
            image: Optional image for vision processing

        Returns:
            Processing results with error information
        """
        print(f"Processing command with safety: '{command}'")

        # Step 1: Safe vision processing
        if image is not None:
            detected_objects = self.safe_vision_processing(image)
        else:
            detected_objects = []
            print("No image provided, skipping vision processing")

        # Step 2: Safe language processing
        action_plan = self.safe_language_processing(command)

        # Step 3: Safe action execution
        if action_plan:
            execution_success = self.safe_action_execution(action_plan)
        else:
            execution_success = False
            print("No action plan generated, skipping execution")

        # Compile results
        results = {
            "command": command,
            "detected_objects": detected_objects,
            "action_plan": action_plan,
            "execution_success": execution_success,
            "error_count": len(self.error_manager.error_log),
            "error_statistics": self.error_manager.get_error_statistics()
        }

        return results

    def run_safety_demo(self):
        """
        Run a demonstration of safe VLA system operation
        """
        print("=== Safe VLA System Demo with Error Handling ===\n")

        # Create a dummy image
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Test commands
        commands = [
            "Pick up the cup",
            "Go to the kitchen",
            "Invalid command that might cause parsing error",
            "Move forward"
        ]

        for command in commands:
            print(f"\n{'='*60}")
            results = self.process_command_with_safety(command, dummy_image)

            print(f"\nCommand: {results['command']}")
            print(f"Objects detected: {len(results['detected_objects'])}")
            print(f"Actions planned: {len(results['action_plan']) if results['action_plan'] else 0}")
            print(f"Execution success: {results['execution_success']}")
            print(f"Errors encountered: {results['error_count']}")

        # Print final error statistics
        stats = self.error_manager.get_error_statistics()
        print(f"\n{'='*60}")
        print("Final Error Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")


def main():
    """
    Main function to demonstrate error handling in VLA systems
    """
    safe_system = SafeVLASystem()
    safe_system.run_safety_demo()


if __name__ == "__main__":
    main()