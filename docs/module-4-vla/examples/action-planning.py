"""
Action Sequence Generation for Cognitive Planning

This example demonstrates how to generate and execute action sequences for robotic tasks
using planning algorithms and state management.
"""

from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Callable
import time
import rospy
from geometry_msgs.msg import Pose, Point
from std_msgs.msg import String
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal


class ActionType(Enum):
    """
    Enum for different types of actions
    """
    NAVIGATION = "navigation"
    MANIPULATION = "manipulation"
    PERCEPTION = "perception"
    WAIT = "wait"
    CONDITIONAL = "conditional"
    SEQUENCE = "sequence"


@dataclass
class Action:
    """
    Data class representing a robot action
    """
    action_type: ActionType
    description: str
    parameters: Dict[str, Any]
    duration_estimate: float = 0.0  # Estimated duration in seconds
    preconditions: List[str] = None
    postconditions: List[str] = None
    success_probability: float = 1.0  # Probability of success (0.0 to 1.0)


class ActionPlanner:
    """
    Class for generating and managing action sequences
    """

    def __init__(self):
        """
        Initialize the action planner
        """
        self.action_library = {}
        self.current_state = {}
        self.history = []

        # Initialize ROS if available
        try:
            rospy.init_node('action_planner', anonymous=True)
            self.ros_available = True
        except:
            self.ros_available = False
            print("ROS not available, running in simulation mode")

    def register_action(self, name: str, action: Action):
        """
        Register an action in the library

        Args:
            name: Name of the action
            action: Action object
        """
        self.action_library[name] = action

    def plan_simple_task(self, command: str) -> List[Action]:
        """
        Plan a simple task based on command

        Args:
            command: Natural language command

        Returns:
            List of actions to execute
        """
        command_lower = command.lower()

        if "pick up" in command_lower or "grasp" in command_lower:
            # Extract object if mentioned
            obj = self._extract_object(command_lower)
            return self._create_pickup_sequence(obj)
        elif "go to" in command_lower or "move to" in command_lower:
            # Extract location if mentioned
            location = self._extract_location(command_lower)
            return self._create_navigation_sequence(location)
        elif "bring" in command_lower or "fetch" in command_lower:
            # Complex task: navigate, pickup, return
            obj = self._extract_object(command_lower)
            return self._create_fetch_sequence(obj)
        else:
            # Default to a simple sequence
            return [
                Action(
                    action_type=ActionType.PERCEPTION,
                    description="Look around to understand environment",
                    parameters={"scan_area": "360_degrees"},
                    duration_estimate=2.0
                ),
                Action(
                    action_type=ActionType.WAIT,
                    description="Wait for user clarification",
                    parameters={"duration": 5.0},
                    duration_estimate=5.0
                )
            ]

    def _extract_object(self, command: str) -> str:
        """
        Extract object name from command (simple implementation)

        Args:
            command: Command string

        Returns:
            Extracted object name
        """
        # Simple keyword matching - in real implementation, use NLP
        objects = ["cup", "bottle", "book", "apple", "box", "ball", "phone"]
        for obj in objects:
            if obj in command:
                return obj
        return "object"

    def _extract_location(self, command: str) -> str:
        """
        Extract location from command (simple implementation)

        Args:
            command: Command string

        Returns:
            Extracted location
        """
        # Simple keyword matching - in real implementation, use NLP
        locations = ["kitchen", "living room", "bedroom", "office", "table", "counter"]
        for loc in locations:
            if loc in command:
                return loc
        return "target_location"

    def _create_pickup_sequence(self, obj: str) -> List[Action]:
        """
        Create a pickup sequence for an object

        Args:
            obj: Object to pickup

        Returns:
            List of actions for pickup
        """
        return [
            Action(
                action_type=ActionType.PERCEPTION,
                description=f"Locate {obj}",
                parameters={"object_type": obj, "search_area": "reachable"},
                duration_estimate=3.0
            ),
            Action(
                action_type=ActionType.NAVIGATION,
                description=f"Approach {obj}",
                parameters={"target_object": obj, "approach_distance": 0.3},
                duration_estimate=5.0
            ),
            Action(
                action_type=ActionType.MANIPULATION,
                description=f"Grasp {obj}",
                parameters={"object_id": obj, "grasp_type": "default"},
                duration_estimate=4.0
            )
        ]

    def _create_navigation_sequence(self, location: str) -> List[Action]:
        """
        Create a navigation sequence to a location

        Args:
            location: Target location

        Returns:
            List of actions for navigation
        """
        return [
            Action(
                action_type=ActionType.NAVIGATION,
                description=f"Navigate to {location}",
                parameters={"destination": location, "speed": "medium"},
                duration_estimate=10.0
            )
        ]

    def _create_fetch_sequence(self, obj: str) -> List[Action]:
        """
        Create a fetch sequence (go, pickup, return)

        Args:
            obj: Object to fetch

        Returns:
            List of actions for fetch task
        """
        # Store current position before going to get the object
        return [
            Action(
                action_type=ActionType.PERCEPTION,
                description=f"Locate {obj}",
                parameters={"object_type": obj, "search_area": "environment"},
                duration_estimate=5.0
            ),
            Action(
                action_type=ActionType.NAVIGATION,
                description=f"Navigate to {obj}",
                parameters={"target_object": obj, "speed": "medium"},
                duration_estimate=8.0
            ),
            Action(
                action_type=ActionType.MANIPULATION,
                description=f"Grasp {obj}",
                parameters={"object_id": obj, "grasp_type": "default"},
                duration_estimate=4.0
            ),
            Action(
                action_type=ActionType.NAVIGATION,
                description="Return to user",
                parameters={"destination": "user_location", "speed": "slow"},
                duration_estimate=10.0
            ),
            Action(
                action_type=ActionType.MANIPULATION,
                description=f"Release {obj} near user",
                parameters={"placement_type": "on_table"},
                duration_estimate=3.0
            )
        ]

    def execute_action_sequence(self, actions: List[Action]) -> Dict[str, Any]:
        """
        Execute a sequence of actions

        Args:
            actions: List of actions to execute

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
            print(f"Executing action {i+1}/{len(actions)}: {action.description}")

            # Log start time
            start_time = time.time()

            # Execute the action
            action_result = self._execute_single_action(action)

            # Calculate duration
            duration = time.time() - start_time
            results["total_duration"] += duration

            # Log execution
            log_entry = {
                "action_index": i,
                "action_description": action.description,
                "duration": duration,
                "success": action_result["success"],
                "details": action_result.get("details", "")
            }
            results["execution_log"].append(log_entry)

            if action_result["success"]:
                results["executed_actions"].append(action)
                print(f"  ✓ Success: {action_result.get('details', 'Action completed')}")
            else:
                results["failed_actions"].append(action)
                results["success"] = False
                print(f"  ✗ Failed: {action_result.get('error', 'Action failed')}")
                # Optionally continue or stop on failure
                break  # Stop on first failure for this example

        return results

    def _execute_single_action(self, action: Action) -> Dict[str, Any]:
        """
        Execute a single action (simulation mode)

        Args:
            action: Action to execute

        Returns:
            Execution result
        """
        # Simulate action execution
        time.sleep(min(action.duration_estimate, 2.0))  # Cap simulation time

        # Simulate success based on success probability
        import random
        success = random.random() < action.success_probability

        if success:
            return {
                "success": True,
                "details": f"Successfully executed {action.action_type.value} action: {action.description}"
            }
        else:
            return {
                "success": False,
                "error": f"Failed to execute {action.action_type.value} action: {action.description}",
                "details": "Action failed due to simulated error"
            }

    def execute_action_sequence_ros(self, actions: List[Action]) -> Dict[str, Any]:
        """
        Execute a sequence of actions using ROS (if available)

        Args:
            actions: List of actions to execute

        Returns:
            Execution results
        """
        if not self.ros_available:
            print("ROS not available, using simulation mode")
            return self.execute_action_sequence(actions)

        results = {
            "success": True,
            "executed_actions": [],
            "failed_actions": [],
            "total_duration": 0.0,
            "execution_log": []
        }

        # Initialize action clients
        move_base_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)

        for i, action in enumerate(actions):
            print(f"Executing action {i+1}/{len(actions)} via ROS: {action.description}")

            start_time = time.time()

            if action.action_type == ActionType.NAVIGATION:
                # Execute navigation action
                success = self._execute_navigation_ros(move_base_client, action)
            elif action.action_type == ActionType.MANIPULATION:
                # Execute manipulation action (placeholder)
                success = self._execute_manipulation_ros(action)
            elif action.action_type == ActionType.PERCEPTION:
                # Execute perception action (placeholder)
                success = self._execute_perception_ros(action)
            elif action.action_type == ActionType.WAIT:
                # Execute wait action
                duration = action.parameters.get("duration", 1.0)
                time.sleep(duration)
                success = True
            else:
                # Default simulation for other action types
                time.sleep(min(action.duration_estimate, 2.0))
                success = True

            duration = time.time() - start_time

            log_entry = {
                "action_index": i,
                "action_description": action.description,
                "duration": duration,
                "success": success,
                "action_type": action.action_type.value
            }
            results["execution_log"].append(log_entry)

            if success:
                results["executed_actions"].append(action)
                print(f"  ✓ Success: {action.description}")
            else:
                results["failed_actions"].append(action)
                results["success"] = False
                print(f"  ✗ Failed: {action.description}")
                break  # Stop on first failure

        return results

    def _execute_navigation_ros(self, client, action: Action) -> bool:
        """
        Execute navigation action using ROS

        Args:
            client: Move base action client
            action: Navigation action

        Returns:
            True if successful, False otherwise
        """
        if not self.ros_available:
            return True  # Simulate success

        # Wait for server
        client.wait_for_server()

        # Create goal
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()

        # Set target pose based on action parameters
        destination = action.parameters.get("destination", "default")
        if destination == "kitchen":
            goal.target_pose.pose.position.x = 1.0
            goal.target_pose.pose.position.y = 2.0
        elif destination == "user_location":
            goal.target_pose.pose.position.x = 0.0
            goal.target_pose.pose.position.y = 0.0
        else:
            goal.target_pose.pose.position.x = 1.0
            goal.target_pose.pose.position.y = 1.0

        goal.target_pose.pose.orientation.w = 1.0

        # Send goal and wait for result
        client.send_goal(goal)
        finished_within_time = client.wait_for_result(rospy.Duration(30))

        return finished_within_time and client.get_state() == actionlib.GoalStatus.SUCCEEDED

    def _execute_manipulation_ros(self, action: Action) -> bool:
        """
        Execute manipulation action using ROS (placeholder)

        Args:
            action: Manipulation action

        Returns:
            True if successful, False otherwise
        """
        # Placeholder for actual manipulation implementation
        print(f"Manipulation action: {action.description}")
        return True

    def _execute_perception_ros(self, action: Action) -> bool:
        """
        Execute perception action using ROS (placeholder)

        Args:
            action: Perception action

        Returns:
            True if successful, False otherwise
        """
        # Placeholder for actual perception implementation
        print(f"Perception action: {action.description}")
        return True


class HierarchicalPlanner:
    """
    Class for hierarchical task planning
    """

    def __init__(self):
        self.planner = ActionPlanner()

    def plan_hierarchical_task(self, high_level_command: str) -> Dict[str, Any]:
        """
        Plan a task hierarchically by decomposing into subtasks

        Args:
            high_level_command: High-level natural language command

        Returns:
            Hierarchical plan
        """
        # Decompose high-level command
        subtasks = self._decompose_command(high_level_command)

        # Plan each subtask
        plan = {
            "original_command": high_level_command,
            "subtasks": [],
            "dependencies": [],
            "estimated_duration": 0.0
        }

        for i, subtask in enumerate(subtasks):
            subtask_actions = self.planner.plan_simple_task(subtask)
            estimated_duration = sum(action.duration_estimate for action in subtask_actions)

            plan["subtasks"].append({
                "id": i,
                "description": subtask,
                "actions": subtask_actions,
                "estimated_duration": estimated_duration
            })

            plan["estimated_duration"] += estimated_duration

        # Determine dependencies (simplified)
        for i in range(len(plan["subtasks"]) - 1):
            plan["dependencies"].append({
                "from": i,
                "to": i + 1,
                "type": "sequential"
            })

        return plan

    def _decompose_command(self, command: str) -> List[str]:
        """
        Decompose a command into subtasks

        Args:
            command: High-level command

        Returns:
            List of subtask descriptions
        """
        command_lower = command.lower()

        if "clean" in command_lower:
            return [
                "Go to the living room",
                "Pick up trash from the floor",
                "Go to the kitchen",
                "Put trash in the bin"
            ]
        elif "set table" in command_lower:
            return [
                "Go to the kitchen",
                "Pick up plates",
                "Go to the dining table",
                "Place plates on the table"
            ]
        elif "bring me" in command_lower or "fetch" in command_lower:
            obj = self.planner._extract_object(command_lower)
            return [
                f"Locate {obj}",
                f"Go to {obj}",
                f"Pick up {obj}",
                "Return to user",
                f"Give {obj} to user"
            ]
        else:
            # Default decomposition
            return [command]


# Example usage
if __name__ == "__main__":
    # Initialize planner
    planner = ActionPlanner()
    hierarchical_planner = HierarchicalPlanner()

    # Test commands
    commands = [
        "Pick up the cup from the table",
        "Go to the kitchen",
        "Bring me the book"
    ]

    print("=== Action Planning Examples ===")

    for command in commands:
        print(f"\nPlanning command: '{command}'")

        # Generate action sequence
        actions = planner.plan_simple_task(command)

        print(f"Generated {len(actions)} actions:")
        for i, action in enumerate(actions, 1):
            print(f"  {i}. {action.action_type.value}: {action.description}")

        # Execute the sequence (simulation)
        print("\nExecuting action sequence...")
        results = planner.execute_action_sequence(actions)

        print(f"Execution completed. Success: {results['success']}")
        print(f"Total duration: {results['total_duration']:.2f}s")
        print(f"Actions executed: {len(results['executed_actions'])}")
        print(f"Actions failed: {len(results['failed_actions'])}")

    print("\n" + "="*50)
    print("=== Hierarchical Planning Examples ===")

    # Test hierarchical planning
    high_level_commands = [
        "Clean the living room",
        "Set the table for dinner",
        "Bring me the book from the bedroom"
    ]

    for command in high_level_commands:
        print(f"\nHierarchical planning for: '{command}'")

        plan = hierarchical_planner.plan_hierarchical_task(command)

        print(f"Decomposed into {len(plan['subtasks'])} subtasks:")
        for i, subtask in enumerate(plan['subtasks']):
            print(f"  {i+1}. {subtask['description']} "
                  f"(~{subtask['estimated_duration']:.1f}s, {len(subtask['actions'])} actions)")

        print(f"Total estimated duration: {plan['estimated_duration']:.1f}s")