---
title: Cognitive Planning with LLMs
sidebar_label: Cognitive Planning with LLMs
sidebar_position: 3
description: Translating natural language into ROS 2 actions using Large Language Models
---

# Cognitive Planning with LLMs

## Introduction

Cognitive planning with Large Language Models (LLMs) represents a revolutionary approach to robot control, where natural language commands are translated into executable action sequences. This chapter explores how LLMs can understand human intent and generate appropriate robot behaviors.

## Natural Language Understanding for Robotics

Natural Language Understanding (NLU) in robotics involves converting high-level human commands into specific robot actions. This requires:

- **Intent Recognition**: Understanding what the user wants to achieve
- **Entity Extraction**: Identifying objects, locations, and parameters
- **Action Sequencing**: Breaking complex commands into executable steps
- **Context Awareness**: Understanding the current environment and robot state

## LLM Integration Architecture

The integration of LLMs into robot control systems typically follows this architecture:

1. **Input Processing**: Preprocess natural language commands
2. **Prompt Engineering**: Format commands for LLM consumption
3. **Action Generation**: Generate action sequences using LLMs
4. **Validation**: Verify generated actions are safe and executable
5. **Execution**: Execute validated actions via ROS 2

## Implementation Example

Here's an example of how to implement LLM-based cognitive planning:

```python
import openai
import rospy
from geometry_msgs.msg import Pose
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import actionlib

class LLMBehaviorPlanner:
    def __init__(self):
        rospy.init_node('llm_behavior_planner')
        self.move_base_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        self.gpt_client = openai.OpenAI()

    def generate_action_plan(self, natural_language_command):
        """Generate action plan from natural language command"""
        prompt = f"""
        Convert the following natural language command into a sequence of robot actions.
        Respond in JSON format with a list of actions. Each action should have:
        - type: 'navigation', 'manipulation', 'perception', or 'other'
        - parameters: relevant parameters for the action

        Command: "{natural_language_command}"

        Example response format:
        {{
            "actions": [
                {{
                    "type": "navigation",
                    "parameters": {{"x": 1.0, "y": 2.0, "theta": 0.0}}
                }},
                {{
                    "type": "perception",
                    "parameters": {{"object_type": "cup"}}
                }},
                {{
                    "type": "manipulation",
                    "parameters": {{"object_id": "cup_1", "action": "grasp"}}
                }}
            ]
        }}
        """

        response = self.gpt_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )

        import json
        action_plan = json.loads(response.choices[0].message.content)
        return action_plan

    def execute_action_plan(self, action_plan):
        """Execute the generated action plan"""
        for action in action_plan['actions']:
            if action['type'] == 'navigation':
                self.execute_navigation(action['parameters'])
            elif action['type'] == 'manipulation':
                self.execute_manipulation(action['parameters'])
            elif action['type'] == 'perception':
                self.execute_perception(action['parameters'])

    def execute_navigation(self, params):
        """Execute navigation action"""
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = params['x']
        goal.target_pose.pose.position.y = params['y']
        goal.target_pose.pose.orientation.z = params['theta']

        self.move_base_client.send_goal_and_wait(goal)
        rospy.loginfo(f"Navigated to ({params['x']}, {params['y']})")

    def execute_manipulation(self, params):
        """Execute manipulation action"""
        # Placeholder for manipulation implementation
        rospy.loginfo(f"Manipulation action: {params['action']} on {params['object_id']}")

    def execute_perception(self, params):
        """Execute perception action"""
        # Placeholder for perception implementation
        rospy.loginfo(f"Perception action: looking for {params['object_type']}")
```

## Task Decomposition and Action Sequencing

Complex commands need to be broken down into simpler, executable actions:

### Example: "Go to the kitchen, pick up the red cup, and bring it to the table"

1. **Navigation**: Move to kitchen area
2. **Perception**: Identify red cup in environment
3. **Manipulation**: Grasp the identified cup
4. **Navigation**: Move to table location
5. **Manipulation**: Release the cup at the table

## State Management and Context Awareness

LLMs need to maintain context about the robot's state and environment:

- **World State**: Current positions of objects and robot
- **Task State**: Progress in current task execution
- **Memory**: Past interactions and learned information
- **Safety Constraints**: Boundaries and limitations

## Prompt Engineering Best Practices

Effective prompt engineering is crucial for reliable LLM-based planning:

- **Clear Instructions**: Specify expected output format
- **Examples**: Provide few-shot examples of correct behavior
- **Constraints**: Define safety and operational boundaries
- **Validation**: Include validation criteria in prompts

## Safety and Validation Considerations

LLM-generated actions must be validated before execution:

- **Safety Filtering**: Prevent unsafe actions
- **Feasibility Check**: Verify actions are physically possible
- **Constraint Validation**: Ensure actions respect operational limits
- **Fallback Mechanisms**: Handle cases where LLM output is invalid

## Multi-step Command Processing

For complex commands requiring multiple steps:

- **Hierarchical Planning**: Break tasks into subtasks
- **Replanning**: Adjust plans based on execution feedback
- **Error Recovery**: Handle failures gracefully
- **Progress Monitoring**: Track task completion

## Performance Considerations

### Latency Optimization

- Cache common command patterns
- Use faster models for simple commands
- Implement streaming responses when possible

### Cost Management

- Optimize prompt length
- Use appropriate model sizes for task complexity
- Implement response caching for common commands

## Integration with ROS 2

The LLM-based planner integrates with ROS 2 through:

- Action servers for complex behaviors
- Services for specific tasks
- Publishers/subscribers for state communication
- TF for coordinate transformations

## Summary

Cognitive planning with LLMs enables robots to understand and execute complex natural language commands by translating high-level intent into executable action sequences. Proper implementation requires careful attention to prompt engineering, validation, and safety considerations.

## Related Topics

To understand the complete Vision-Language-Action pipeline, explore these related chapters:
- [Voice-to-Action Systems](./voice-to-action.md) - Learn how speech input is processed and converted to robot commands using OpenAI Whisper
- [Vision-Guided Manipulation](./vision-guided-manipulation.md) - Discover how computer vision enables robots to interact with objects in their environment
- [Multimodal Fusion Techniques](./multimodal-fusion.md) - Explore how voice, vision, and planning components are combined in VLA systems
- [VLA Pipeline Integration](./integration.md) - Understand how all VLA components work together in a unified system