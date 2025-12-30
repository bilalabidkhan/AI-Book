"""
LLM Prompt Engineering for Cognitive Planning

This example demonstrates effective prompt engineering techniques for LLM-based
robotic planning and control.
"""

import openai
import json
from typing import Dict, List, Any
import re


class VLAPromptEngineer:
    """
    Class for engineering effective prompts for Vision-Language-Action systems
    """

    def __init__(self, client=None):
        """
        Initialize the prompt engineer

        Args:
            client: OpenAI client (or other LLM client)
        """
        self.client = client

    def create_robot_control_prompt(self, natural_language_command: str, robot_capabilities: List[str],
                                   environment_context: Dict[str, Any] = None) -> str:
        """
        Create a structured prompt for robot control

        Args:
            natural_language_command: Natural language command from user
            robot_capabilities: List of robot capabilities
            environment_context: Context about the current environment

        Returns:
            Formatted prompt string
        """
        capabilities_str = ", ".join(robot_capabilities)

        environment_context_str = ""
        if environment_context:
            env_items = [f"{key}: {value}" for key, value in environment_context.items()]
            environment_context_str = "Environment Context:\n" + "\n".join([f"  - {item}" for item in env_items])

        prompt = f"""
You are a robot command interpreter. Convert the following natural language command into a sequence of specific robot actions.

Robot Capabilities: {capabilities_str}

{environment_context_str}

Command: "{natural_language_command}"

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
    "safety_check": "Brief safety assessment",
    "estimated_steps": "Number of actions needed"
}}

Example response for "Go to the kitchen and bring me a cup":
{{
    "intent": "Retrieve cup from kitchen",
    "action_sequence": [
        {{
            "action_type": "navigation",
            "description": "Navigate to kitchen area",
            "parameters": {{"destination": "kitchen", "speed": "medium"}}
        }},
        {{
            "action_type": "perception",
            "description": "Look for cup in kitchen",
            "parameters": {{"object_type": "cup", "search_area": "countertops"}}
        }},
        {{
            "action_type": "manipulation",
            "description": "Grasp the cup",
            "parameters": {{"object_id": "detected_cup", "grasp_type": "top_grasp"}}
        }},
        {{
            "action_type": "navigation",
            "description": "Return to user",
            "parameters": {{"destination": "user_location", "speed": "slow"}}
        }},
        {{
            "action_type": "manipulation",
            "description": "Place cup near user",
            "parameters": {{"placement_type": "on_table"}}
        }}
    ],
    "safety_check": "Path is clear, cup is not fragile",
    "estimated_steps": 5
}}
"""

        return prompt.strip()

    def create_spatial_reasoning_prompt(self, command: str, object_locations: Dict[str, Any]) -> str:
        """
        Create a prompt for spatial reasoning tasks

        Args:
            command: Natural language command involving spatial relationships
            object_locations: Dictionary of object locations

        Returns:
            Formatted prompt string
        """
        locations_str = json.dumps(object_locations, indent=2)

        prompt = f"""
You are a spatial reasoning assistant for robotics. Interpret the spatial relationships in the following command.

Object Locations:
{locations_str}

Command: "{command}"

Analyze the spatial relationships and provide a JSON response with:
- target_object: The object to be manipulated
- reference_object: The object used for spatial reference
- relationship: The spatial relationship (left, right, near, far, on, under, etc.)
- coordinates: Estimated coordinates for navigation or manipulation

Response format:
{{
    "target_object": "object name",
    "reference_object": "object name",
    "relationship": "spatial relationship",
    "spatial_action": "what the robot should do",
    "estimated_coordinates": {{"x": 0.0, "y": 0.0, "z": 0.0}}
}}
"""

        return prompt.strip()

    def create_multi_step_planning_prompt(self, complex_command: str, robot_state: Dict[str, Any]) -> str:
        """
        Create a prompt for complex multi-step planning

        Args:
            complex_command: Complex multi-step command
            robot_state: Current state of the robot

        Returns:
            Formatted prompt string
        """
        state_str = json.dumps(robot_state, indent=2)

        prompt = f"""
You are a multi-step planning assistant for robotics. Break down the following complex command into a sequence of achievable steps.

Current Robot State:
{state_str}

Command: "{complex_command}"

Provide a detailed plan with the following considerations:
1. Task decomposition: Break into subtasks
2. Prerequisites: What needs to be done before each step
3. Dependencies: Which steps depend on others
4. Safety: Potential risks and mitigation strategies

Response format:
{{
    "task_breakdown": [
        {{
            "step": 1,
            "subtask": "Description of subtask",
            "action_type": "navigation|manipulation|perception|other",
            "prerequisites": ["list", "of", "prerequisites"],
            "estimated_duration": "time estimate",
            "safety_considerations": ["list", "of", "safety", "items"]
        }}
    ],
    "overall_strategy": "High-level strategy",
    "fallback_plans": ["list", "of", "fallback", "options"]
}}
"""

        return prompt.strip()

    def execute_command_with_llm(self, command: str, capabilities: List[str],
                                environment: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute a command by sending it to an LLM and parsing the response

        Args:
            command: Natural language command
            capabilities: Robot capabilities
            environment: Environment context

        Returns:
            Parsed response from the LLM
        """
        if not self.client:
            # For demonstration, return a mock response
            return self._mock_llm_response(command)

        prompt = self.create_robot_control_prompt(command, capabilities, environment)

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1000
            )

            # Extract JSON from response
            response_text = response.choices[0].message.content
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)

            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
            else:
                raise ValueError("No JSON found in response")
        except Exception as e:
            print(f"Error calling LLM: {e}")
            return self._mock_llm_response(command)

    def _mock_llm_response(self, command: str) -> Dict[str, Any]:
        """
        Mock LLM response for demonstration purposes

        Args:
            command: Command to mock response for

        Returns:
            Mock response
        """
        # Simple mock responses based on command keywords
        if "move" in command.lower() or "go" in command.lower():
            return {
                "intent": f"Move as requested: {command}",
                "action_sequence": [
                    {
                        "action_type": "navigation",
                        "description": f"Navigate based on command: {command}",
                        "parameters": {"destination": "target_location", "speed": "medium"}
                    }
                ],
                "safety_check": "Basic safety check passed",
                "estimated_steps": 1
            }
        elif "pick" in command.lower() or "grasp" in command.lower():
            return {
                "intent": f"Manipulation task: {command}",
                "action_sequence": [
                    {
                        "action_type": "perception",
                        "description": "Locate target object",
                        "parameters": {"object_type": "target_object"}
                    },
                    {
                        "action_type": "manipulation",
                        "description": f"Grasp object as requested: {command}",
                        "parameters": {"grasp_type": "pinch_grasp"}
                    }
                ],
                "safety_check": "Manipulation safety check passed",
                "estimated_steps": 2
            }
        else:
            return {
                "intent": f"General task: {command}",
                "action_sequence": [
                    {
                        "action_type": "other",
                        "description": f"Process command: {command}",
                        "parameters": {}
                    }
                ],
                "safety_check": "General safety check passed",
                "estimated_steps": 1
            }


class SafetyFilter:
    """
    Class to filter LLM responses for safety
    """

    def __init__(self):
        self.dangerous_actions = [
            "jump", "run fast", "collide", "crash", "break", "damage",
            "harm", "injure", "unsafe", "dangerous", "risk"
        ]
        self.forbidden_objects = [
            "human", "person", "face", "body", "head", "hand", "arm"
        ]

    def filter_response(self, llm_response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter LLM response for safety

        Args:
            llm_response: Response from LLM

        Returns:
            Filtered response with safety considerations
        """
        filtered_response = llm_response.copy()

        # Check action sequence for dangerous actions
        safe_actions = []
        for action in llm_response.get("action_sequence", []):
            is_safe = True

            # Check for dangerous keywords in description
            desc = action.get("description", "").lower()
            if any(danger in desc for danger in self.dangerous_actions):
                is_safe = False

            # Check for forbidden objects
            if any(forbidden in desc for forbidden in self.forbidden_objects):
                is_safe = False

            if is_safe:
                safe_actions.append(action)
            else:
                print(f"⚠️  Filtering unsafe action: {action['description']}")

        filtered_response["action_sequence"] = safe_actions
        filtered_response["safety_filtered"] = len(llm_response.get("action_sequence", [])) - len(safe_actions)

        return filtered_response


# Example usage
if __name__ == "__main__":
    # Initialize the prompt engineer
    engineer = VLAPromptEngineer()

    # Define robot capabilities
    capabilities = [
        "navigation", "object manipulation", "grasping", "perception",
        "object recognition", "path planning", "grasp planning"
    ]

    # Define environment context
    environment = {
        "room_layout": "kitchen with counter, table, and chairs",
        "objects_present": ["cup", "plate", "bottle", "apple"],
        "robot_position": {"x": 0.0, "y": 0.0, "z": 0.0},
        "user_position": {"x": 2.0, "y": 1.0, "z": 0.0}
    }

    # Test commands
    commands = [
        "Go to the kitchen and bring me a cup",
        "Pick up the red apple from the table",
        "Move the book to the left of the lamp"
    ]

    # Process each command
    for command in commands:
        print(f"\nProcessing command: {command}")

        # Get LLM response
        response = engineer.execute_command_with_llm(command, capabilities, environment)

        # Apply safety filtering
        safety_filter = SafetyFilter()
        safe_response = safety_filter.filter_response(response)

        print(f"Intent: {safe_response['intent']}")
        print(f"Estimated steps: {safe_response['estimated_steps']}")
        print(f"Safety check: {safe_response['safety_check']}")
        print("Action sequence:")
        for i, action in enumerate(safe_response['action_sequence'], 1):
            print(f"  {i}. {action['action_type']}: {action['description']}")
        print("-" * 50)