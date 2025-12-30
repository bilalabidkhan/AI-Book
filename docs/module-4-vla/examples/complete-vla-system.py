"""
Complete Vision-Language-Action (VLA) System

This example demonstrates the complete integration of vision, language, and action
components in a unified system for robot control.
"""

import openai
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import time
import json
import re


@dataclass
class VLAState:
    """
    Data class representing the state of the VLA system
    """
    robot_position: Dict[str, float] = None
    detected_objects: List[Dict[str, Any]] = None
    current_task: str = ""
    task_history: List[str] = None
    user_preferences: Dict[str, Any] = None

    def __post_init__(self):
        if self.robot_position is None:
            self.robot_position = {"x": 0.0, "y": 0.0, "z": 0.0}
        if self.detected_objects is None:
            self.detected_objects = []
        if self.task_history is None:
            self.task_history = []
        if self.user_preferences is None:
            self.user_preferences = {}


class VisionSystem:
    """
    Vision component of the VLA system
    """

    def __init__(self, confidence_threshold: float = 0.5):
        """
        Initialize the vision system
        """
        self.weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = fasterrcnn_resnet50_fpn(weights=self.weights)
        self.model.eval()
        self.confidence_threshold = confidence_threshold
        self.category_names = self.weights.meta["categories"]
        self.transform = T.Compose([T.ToTensor()])

    def detect_objects(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect objects in an image

        Args:
            image: Input image as numpy array

        Returns:
            List of detected objects
        """
        # Preprocess image
        image_tensor = self.transform(image)
        image_tensor = image_tensor.unsqueeze(0)

        # Perform inference
        with torch.no_grad():
            predictions = self.model(image_tensor)

        # Process results
        boxes = predictions[0]['boxes'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()

        # Filter detections
        valid_detections = scores > self.confidence_threshold

        detected_objects = []
        for i, valid in enumerate(valid_detections):
            if valid:
                obj = {
                    'id': f"obj_{i}",
                    'label': self.category_names[labels[i]],
                    'confidence': float(scores[i]),
                    'bbox': [float(x) for x in boxes[i]],  # [x1, y1, x2, y2]
                    'center': [(boxes[i][0] + boxes[i][2]) / 2, (boxes[i][1] + boxes[i][3]) / 2]
                }
                detected_objects.append(obj)

        return detected_objects

    def get_object_by_label(self, objects: List[Dict[str, Any]], label: str) -> Optional[Dict[str, Any]]:
        """
        Get an object by its label

        Args:
            objects: List of detected objects
            label: Object label to find

        Returns:
            Detected object or None
        """
        for obj in objects:
            if obj['label'].lower() == label.lower():
                return obj
        return None

    def get_closest_object(self, objects: List[Dict[str, Any]], target_center: List[float]) -> Optional[Dict[str, Any]]:
        """
        Get the closest object to a target center

        Args:
            objects: List of detected objects
            target_center: Target center [x, y]

        Returns:
            Closest object or None
        """
        if not objects:
            return None

        closest_obj = None
        min_distance = float('inf')

        for obj in objects:
            center = obj['center']
            distance = np.sqrt((center[0] - target_center[0])**2 + (center[1] - target_center[1])**2)
            if distance < min_distance:
                min_distance = distance
                closest_obj = obj

        return closest_obj


class LanguageSystem:
    """
    Language component of the VLA system
    """

    def __init__(self, client=None):
        """
        Initialize the language system
        """
        self.client = client
        self.command_keywords = {
            'navigation': ['go to', 'move to', 'navigate to', 'walk to', 'travel to'],
            'manipulation': ['pick up', 'grasp', 'take', 'grab', 'lift', 'hold'],
            'perception': ['find', 'locate', 'look for', 'search for', 'see'],
            'placement': ['put', 'place', 'set down', 'release', 'drop']
        }

    def parse_command(self, command: str) -> Dict[str, Any]:
        """
        Parse a natural language command

        Args:
            command: Natural language command

        Returns:
            Parsed command structure
        """
        command_lower = command.lower()
        parsed = {
            'action_type': 'unknown',
            'target_object': None,
            'target_location': None,
            'command': command,
            'action_sequence': []
        }

        # Identify action type based on keywords
        for action_type, keywords in self.command_keywords.items():
            if any(keyword in command_lower for keyword in keywords):
                parsed['action_type'] = action_type
                break

        # Extract target object
        objects = ['cup', 'bottle', 'book', 'apple', 'box', 'plate', 'phone', 'bowl']
        for obj in objects:
            if obj in command_lower:
                parsed['target_object'] = obj
                break

        # Extract target location
        locations = ['kitchen', 'living room', 'bedroom', 'office', 'table', 'counter', 'couch', 'chair']
        for loc in locations:
            if loc in command_lower:
                parsed['target_location'] = loc
                break

        return parsed

    def generate_action_plan(self, command: str, detected_objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate an action plan for a command using LLM

        Args:
            command: Natural language command
            detected_objects: List of detected objects from vision system

        Returns:
            List of actions to execute
        """
        if not self.client:
            # Use rule-based planning for demonstration
            return self._rule_based_planning(command, detected_objects)

        # Create prompt for LLM
        objects_str = json.dumps(detected_objects[:5], indent=2)  # Limit to first 5 objects
        prompt = f"""
Convert the following natural language command into a sequence of robot actions.

Current environment objects:
{objects_str}

Command: "{command}"

Provide your response as a JSON object with the following structure:
{{
    "intent": "Brief description of the user's intent",
    "action_sequence": [
        {{
            "action_type": "navigation|manipulation|perception|other",
            "description": "What the robot should do",
            "parameters": {{"key": "value"}}
        }}
    ],
    "safety_check": "Brief safety assessment"
}}
"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500
            )

            # Extract JSON from response
            response_text = response.choices[0].message.content
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)

            if json_match:
                json_str = json_match.group()
                result = json.loads(json_str)
                return result.get('action_sequence', [])
            else:
                raise ValueError("No JSON found in response")
        except Exception as e:
            print(f"Error calling LLM: {e}")
            return self._rule_based_planning(command, detected_objects)

    def _rule_based_planning(self, command: str, detected_objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Fallback rule-based planning

        Args:
            command: Natural language command
            detected_objects: List of detected objects

        Returns:
            List of actions to execute
        """
        command_lower = command.lower()
        actions = []

        if 'pick up' in command_lower or 'grasp' in command_lower or 'take' in command_lower:
            # Look for target object
            target_obj = None
            for obj in detected_objects:
                if obj['label'] in command_lower:
                    target_obj = obj
                    break

            if target_obj:
                actions.extend([
                    {
                        "action_type": "navigation",
                        "description": f"Navigate to {target_obj['label']}",
                        "parameters": {"target_object_id": target_obj['id']}
                    },
                    {
                        "action_type": "manipulation",
                        "description": f"Grasp {target_obj['label']}",
                        "parameters": {"object_id": target_obj['id']}
                    }
                ])
            else:
                actions.append({
                    "action_type": "perception",
                    "description": f"Look for requested object in command: {command}",
                    "parameters": {"search_area": "environment"}
                })

        elif 'go to' in command_lower or 'move to' in command_lower:
            location = self._extract_location(command_lower)
            actions.append({
                "action_type": "navigation",
                "description": f"Navigate to {location}",
                "parameters": {"destination": location}
            })

        else:
            actions.append({
                "action_type": "perception",
                "description": "Look around to understand environment",
                "parameters": {"scan_area": "360_degrees"}
            })

        return actions

    def _extract_location(self, command: str) -> str:
        """
        Extract location from command (simple implementation)

        Args:
            command: Command string

        Returns:
            Extracted location
        """
        locations = ["kitchen", "living room", "bedroom", "office", "table", "counter", "couch", "chair"]
        for loc in locations:
            if loc in command:
                return loc
        return "target_location"


class ActionSystem:
    """
    Action component of the VLA system
    """

    def __init__(self):
        """
        Initialize the action system
        """
        self.execution_history = []

    def execute_action_sequence(self, actions: List[Dict[str, Any]], state: VLAState) -> Dict[str, Any]:
        """
        Execute a sequence of actions

        Args:
            actions: List of actions to execute
            state: Current VLA state

        Returns:
            Execution results
        """
        results = {
            "success": True,
            "executed_actions": [],
            "failed_actions": [],
            "total_duration": 0.0,
            "execution_log": []
        }

        for i, action in enumerate(actions):
            print(f"Executing action {i+1}/{len(actions)}: {action['description']}")

            start_time = time.time()

            # Execute the action based on type
            action_result = self._execute_single_action(action, state)

            duration = time.time() - start_time
            results["total_duration"] += duration

            log_entry = {
                "action_index": i,
                "action_description": action['description'],
                "duration": duration,
                "success": action_result["success"],
                "action_type": action.get('action_type', 'unknown')
            }
            results["execution_log"].append(log_entry)

            if action_result["success"]:
                results["executed_actions"].append(action)
                print(f"  ✓ Success: {action_result.get('details', 'Action completed')}")
            else:
                results["failed_actions"].append(action)
                results["success"] = False
                print(f"  ✗ Failed: {action_result.get('error', 'Action failed')}")
                break  # Stop on first failure

        return results

    def _execute_single_action(self, action: Dict[str, Any], state: VLAState) -> Dict[str, Any]:
        """
        Execute a single action

        Args:
            action: Action to execute
            state: Current VLA state

        Returns:
            Execution result
        """
        # Simulate action execution
        action_type = action.get('action_type', 'unknown')
        duration = action.get('parameters', {}).get('duration', 1.0)

        # Simulate different action types
        if action_type == 'navigation':
            # Simulate navigation
            time.sleep(0.5)  # Simulated duration
            state.robot_position = {"x": np.random.uniform(-5, 5), "y": np.random.uniform(-5, 5), "z": 0.0}
        elif action_type == 'manipulation':
            # Simulate manipulation
            time.sleep(0.7)
        elif action_type == 'perception':
            # Simulate perception
            time.sleep(0.3)
        else:
            # Other action types
            time.sleep(duration)

        return {
            "success": True,
            "details": f"Successfully executed {action_type} action: {action['description']}"
        }


class VLASystem:
    """
    Complete Vision-Language-Action system
    """

    def __init__(self, openai_client=None):
        """
        Initialize the VLA system

        Args:
            openai_client: OpenAI client for LLM integration
        """
        self.vision_system = VisionSystem()
        self.language_system = LanguageSystem(openai_client)
        self.action_system = ActionSystem()
        self.state = VLAState()

    def process_command(self, command: str, camera_image: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Process a complete VLA command

        Args:
            command: Natural language command
            camera_image: Optional camera image for vision processing

        Returns:
            Complete processing results
        """
        print(f"Processing VLA command: '{command}'")

        # Step 1: Update state with current vision data if available
        if camera_image is not None:
            detected_objects = self.vision_system.detect_objects(camera_image)
            self.state.detected_objects = detected_objects
            print(f"Detected {len(detected_objects)} objects in environment")

        # Step 2: Parse command and generate action plan
        parsed_command = self.language_system.parse_command(command)
        action_plan = self.language_system.generate_action_plan(command, self.state.detected_objects)

        print(f"Generated action plan with {len(action_plan)} steps")

        # Step 3: Execute action plan
        execution_results = self.action_system.execute_action_sequence(action_plan, self.state)

        # Update state with task history
        self.state.task_history.append(command)

        # Compile results
        results = {
            "command": command,
            "parsed_command": parsed_command,
            "action_plan": action_plan,
            "execution_results": execution_results,
            "final_state": self.state,
            "success": execution_results["success"]
        }

        return results

    def run_demo(self):
        """
        Run a demonstration of the VLA system
        """
        print("=== Vision-Language-Action (VLA) System Demo ===\n")

        # Create a dummy camera image for demonstration
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Add some simulated objects to the image
        cv2.rectangle(dummy_image, (100, 100), (150, 150), (255, 0, 0), 2)  # Blue box (could be a book)
        cv2.circle(dummy_image, (300, 200), 25, (0, 255, 0), 2)  # Green circle (could be an apple)
        cv2.rectangle(dummy_image, (400, 300), (450, 350), (0, 0, 255), 2)  # Red box (could be a cup)

        # Test commands
        commands = [
            "Pick up the cup",
            "Go to the kitchen",
            "Find the apple",
            "Move to the table and look around"
        ]

        for command in commands:
            print(f"\n{'='*60}")
            results = self.process_command(command, dummy_image)

            print(f"\nCommand: {results['command']}")
            print(f"Action Plan: {len(results['action_plan'])} steps")
            print(f"Execution Success: {results['success']}")
            print(f"Execution Duration: {results['execution_results']['total_duration']:.2f}s")

            if results['success']:
                print("✓ Task completed successfully")
            else:
                print("✗ Task failed")

        print(f"\n{'='*60}")
        print("Demo completed!")
        print(f"Robot position: {self.state.robot_position}")
        print(f"Objects detected in last frame: {len(self.state.detected_objects)}")
        print(f"Tasks completed: {len(self.state.task_history)}")


def main():
    """
    Main function to run the VLA system
    """
    # Initialize VLA system (without OpenAI client for demo)
    vla_system = VLASystem()

    # Run demonstration
    vla_system.run_demo()


if __name__ == "__main__":
    main()