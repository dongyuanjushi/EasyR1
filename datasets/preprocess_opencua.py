from ast import main
import os
import json
import uuid
import jsonlines
from transformers import AutoTokenizer
from datetime import datetime

trajectory_dir = "datasets/opencua-raw"

tools_def = [
    {
        "type": "function",
        "function": {
            "name": "key",
            "description": "Performs key down presses on the arguments passed in order, then performs key releases in reverse order.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keys": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of keys to press in order (e.g., ['ctrl', 'c'])"
                    }
                },
                "required": ["keys"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "type",
            "description": "Type a string of text on the keyboard.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text string to type"
                    }
                },
                "required": ["text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "mouse_move",
            "description": "Move the cursor to a specified (x, y) pixel coordinate on the screen.",
            "parameters": {
                "type": "object",
                "properties": {
                    "coordinate": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 2,
                        "maxItems": 2,
                        "description": "The [x, y] pixel coordinates to move to"
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "left_click",
            "description": "Click the left mouse button at a specified (x, y) pixel coordinate on the screen.",
            "parameters": {
                "type": "object",
                "properties": {
                    "coordinate": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 2,
                        "maxItems": 2,
                        "description": "The [x, y] pixel coordinates to click"
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "left_click_drag",
            "description": "Click and drag the cursor to a specified (x, y) pixel coordinate on the screen.",
            "parameters": {
                "type": "object",
                "properties": {
                    "coordinate": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 2,
                        "maxItems": 2,
                        "description": "The [x, y] pixel coordinates to drag to"
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "right_click",
            "description": "Click the right mouse button at a specified (x, y) pixel coordinate on the screen.",
            "parameters": {
                "type": "object",
                "properties": {
                    "coordinate": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 2,
                        "maxItems": 2,
                        "description": "The [x, y] pixel coordinates to right-click"
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "middle_click",
            "description": "Click the middle mouse button at a specified (x, y) pixel coordinate on the screen.",
            "parameters": {
                "type": "object",
                "properties": {
                    "coordinate": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 2,
                        "maxItems": 2,
                        "description": "The [x, y] pixel coordinates to middle-click"
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "double_click",
            "description": "Double-click the left mouse button at a specified (x, y) pixel coordinate on the screen.",
            "parameters": {
                "type": "object",
                "properties": {
                    "coordinate": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 2,
                        "maxItems": 2,
                        "description": "The [x, y] pixel coordinates to double-click"
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "triple_click",
            "description": "Triple-click the left mouse button at a specified (x, y) pixel coordinate on the screen (simulated as double-click since it's the closest action).",
            "parameters": {
                "type": "object",
                "properties": {
                    "coordinate": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 2,
                        "maxItems": 2,
                        "description": "The [x, y] pixel coordinates to triple-click"
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "scroll",
            "description": "Performs a scroll of the mouse scroll wheel.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pixels": {
                        "type": "number",
                        "description": "The amount of pixels to scroll (positive for down, negative for up)"
                    }
                },
                "required": ["pixels"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "hscroll",
            "description": "Performs a horizontal scroll (mapped to regular scroll).",
            "parameters": {
                "type": "object",
                "properties": {
                    "pixels": {
                        "type": "number",
                        "description": "The amount of pixels to scroll horizontally"
                    }
                },
                "required": ["pixels"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "wait",
            "description": "Wait specified seconds for the change to happen.",
            "parameters": {
                "type": "object",
                "properties": {
                    "time": {
                        "type": "number",
                        "description": "The number of seconds to wait"
                    }
                },
                "required": ["time"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "terminate",
            "description": "Terminate the current task and report its completion status.",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["success", "failure"],
                        "description": "The status of the task completion"
                    }
                },
                "required": ["status"]
            }
        }
    }
]


ACTION_N = 1

SYSTEM_PROMPT_POLICY_MODEL_ACTION_ONLY = f"""You are a helpful assistant that can understand screenshots in the images and take actions in a computer environment to achieve the task.
## Tools

You may call the tools defined below to assist with the given task.

Here are some tips for using the tools:
- Use a mouse and keyboard to interact with a computer, and take screenshots.",
- This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.",
- Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions. E.g. if you click on Firefox and a window doesn't open, try wait and taking another screenshot.",
- Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.",
- If you tried clicking on a program or link but it failed to load even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.",
- Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.",

You are provided with function signatures within <tools></tools> XML tags:
<tools>
""" + json.dumps(tools_def) + """
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

## Response format

Response format for every step:
1) A single <action>...</action> block containing only a short imperative describing what to do in the UI.
2) A single <tool_call>...</tool_call> block containing only the JSON: {"name": <function-name>, "arguments": <args-json-object>}. 
""" + f"""## Action Rules:
- The system is running on a x86_64 ubuntu system.
- Chrome is the default browser that have been installed for you to use.
- The current working directory is /home/user.
- The password for the user is "password". Use it when you need to authenticate or use sudo commands.
- The current date is {datetime.now().strftime("%Y-%m-%d")}.
- Execute exactly one action per interaction.
- Output exactly in the order: Action, <tool_call>.
- Be brief: one sentence for Action.
- Do not output anything else outside those parts.
- Leave all windows and applications open after completing the task.
- If finishing, use action=terminate in the tool call. 
    - Issue the status as success if the task is completed successfully
    - Issue the status as failure if the task is infeasible to complete due to environment constraints. 
"""

SYSTEM_PROMPT_POLICY_MODEL_OBSERVATION_ACTION = f"""You are a helpful assistant that can understand screenshots in the images and take actions in a computer environment to achieve the task. 
The screenshot resolution is 1000x1000. Make sure the coordinates of your actions are within the screen resolution.
## Tools

You may call the tools defined below to assist with the given task.

Here are some tips for using the tools:
- Use a mouse and keyboard to interact with a computer, and take screenshots.",
- This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.",
- Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions. E.g. if you click on Firefox and a window doesn't open, try wait and taking another screenshot.",
- Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.",
- If you tried clicking on a program or link but it failed to load even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.",
- Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.",

You are provided with function signatures within <tools></tools> XML tags:
<tools>
""" + json.dumps(tools_def) + """
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

## Response format

Response format for every step:
1) A single <observation>...</observation> block containing what you have observed in the current state of the UI. Focus on the changes from previous states. 
2) A single <action>...</action> block containing only a short imperative describing what to do in the UI.
3) A single <tool_call>...</tool_call> block containing only the JSON: {"name": <function-name>, "arguments": <args-json-object>}. 
""" + f"""## Action Rules:
- The system is running on a x86_64 ubuntu system.
- The screenshot resolution is 1000x1000. Make sure the coordinates of your actions are within the screen resolution.
- Chrome is the default browser that have been installed for you to use.
- The current working directory is /home/user.
- The password for the user is "password". Use it when you need to authenticate or use sudo commands.
- The current date is {datetime.now().strftime("%Y-%m-%d")}.
- Execute exactly one action per interaction.
- Output exactly in the order: Action, <tool_call>.
- Be brief: one sentence for Action.
- Do not output anything else outside those parts.
- Leave all windows and applications open after completing the task.
- If finishing, use action=terminate in the tool call. 
    - Issue the status as success if the task is completed successfully
    - Issue the status as failure if the task is infeasible to complete due to environment constraints. 
"""


def load_successful_trajectories(trajectory_dir):
    domain_map = {
        "vscode": "vs_code",
        "multiapp": "multi_app",
        "libreoffice_impress": "libreoffice_impress",
        "libreoffice_writer": "libreoffice_writer",
        "libreoffice_calc": "libreoffice_calc",
        "chrome": "chrome",
        "gimp": "gimp",
        "thunderbird": "thunderbird",
        "vlc": "vlc",
        "os": "os"
    }
    successful_trajectories = []

    domain_cnt = {
        "vs_code": 0,
        "multi_app": 0,
        "chrome": 0,
        "libreoffice_impress": 0,
        "libreoffice_writer": 0,
        "libreoffice_calc": 0,
        "gimp": 0,
        "thunderbird": 0,
        "vlc": 0,
        "os": 0
    }

    with jsonlines.open(os.path.join(trajectory_dir, "agentnet_ubuntu_5k.jsonl")) as reader:
        for trajectory in reader:
            if "task_completed" not in trajectory:
                continue
            
            if trajectory["task_completed"]:
                domain = trajectory["domain"].lower()
                if domain not in domain_map:
                    continue
                domain = domain_map[domain]
                domain_cnt[domain] += 1
                if domain_cnt[domain] >= 30:
                    continue
                else:
                    instruction = trajectory["instruction"]
                    id = trajectory["task_id"]
                    traj = trajectory["traj"]
                    successful_trajectories.append({
                        "id": id,
                        "instruction": instruction,
                        "domain": domain,
                        "traj": traj
                    })
            
    return successful_trajectories

def map_assistant_response(action, action_code):
    tool_call = map_action_code_to_tool_call(action_code)
    return f"<action>{action}</action>\n{tool_call}"

# Gather and print all unique action types found in all successful trajectories.

def construct_action_only_samples(trajectory):
    all_samples = []
    instruction = trajectory["instruction"]
    last_k = 2
    length = len(trajectory["traj"])

    start_idx = 0
    end_idx = length

    for i in range(start_idx, end_idx):
        history_messages = []
        history_images = []
        action_history = []

        prev_start = max(0, i - last_k)
        if trajectory["traj"][i]["value"]["last_step_correct"] is False:
            continue

        for j in range(prev_start, i):
            prev_image_id = trajectory["traj"][j]["image"]
            prev_img_path = os.path.join(trajectory_dir, "ubuntu_images", f"{prev_image_id}")

            history_images.append(prev_img_path)
        
        for j in range(prev_start, i):
            action = trajectory["traj"][j]["value"]["action"]
            action_code = trajectory["traj"][j]["value"]["code"]
            action_only_response = map_assistant_response(action, action_code)
            action_history.append(action_only_response)

        assert len(action_history) == len(history_images)

        for i in range(len(action_history)):
            image_message = {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "<image>"
                    }
                ]
            }

            action_message = {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": action_history[i]
                    }
                ]
            }

            history_messages.append(image_message)
            history_messages.append(action_message)

        current_image_id = trajectory["traj"][i]["image"]
        current_img_path = os.path.join(trajectory_dir, "ubuntu_images", f"{current_image_id}")

        current_action = trajectory["traj"][i]["value"]["action"]
        current_action_code = trajectory["traj"][i]["value"]["code"]
        current_action_only_output = map_assistant_response(current_action, current_action_code)

        current_image_message = {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "<image>"
                }
            ]
        }

        history_messages.append(current_image_message)
        # history_messages.append(current_action_message)

        history_images.append(current_img_path)
        
        user_instruction = f"Overall task instruction: {instruction}."
        
        data_sample = {
            "system": SYSTEM_PROMPT_POLICY_MODEL_ACTION_ONLY,
            "instruction": user_instruction,
            "history_messages": history_messages,
            "output": current_action_only_output,
            "images": history_images
        }
        all_samples.append(data_sample)
    
    return all_samples


def construct_samples(trajectory):
    data_samples = construct_action_only_samples(trajectory)
    return data_samples

def map_action_code_to_tool_call(action_code):
    """
    Maps action code strings from the original trajectories (e.g., pyautogui.* commands)
    to the tool_call XML format defined in tools_def.

    Parameters:
        action_code (str): Action code like "pyautogui.click(x=123, y=456)"
    
    Returns:
        str: XML string of the form <tool_call>{"name": ..., "arguments": {...}}</tool_call>
    """
    import re

    # Regexes for basic pyautogui actions used in AgentNet-formatted trajectories
    # Map pyautogui.hotkey (and pyautogui.hotkey_all) to key action.
    hotkey_patterns = [
        (r"pyautogui\.hotkey\(\s*(.*?)\s*\)", False)
    ]
    for pattern, _ in hotkey_patterns:
        match = re.search(pattern, action_code)
        if match:
            keys_str = match.group(1)
            # Support both 'ctrl', 'alt', 'x' and also ctrl, alt, x without quotes
            keys = []
            for k in keys_str.split(","):
                k = k.strip()
                if k.startswith("'") or k.startswith('"'):
                    k = k.strip("'\"")  # remove quotes
                keys.append(k)
            args = {"keys": keys}
            tool_call_obj = {
                "name": "key",
                "arguments": args
            }
            return f'<tool_call>{json.dumps(tool_call_obj, ensure_ascii=False)}</tool_call>'

    patterns = {
        "left_click": r"pyautogui\.click\(x\s*=\s*([-+]?\d*\.\d+|\d+),\s*y\s*=\s*([-+]?\d*\.\d+|\d+)\)",
        "double_click": r"pyautogui\.doubleClick\(x\s*=\s*([-+]?\d*\.\d+|\d+),\s*y\s*=\s*([-+]?\d*\.\d+|\d+)\)",
        "triple_click": r"computer\.tripleClick\(x\s*=\s*([-+]?\d*\.\d+|\d+),\s*y\s*=\s*([-+]?\d*\.\d+|\d+)\)",
        "mouse_move": r"pyautogui\.moveTo\(x\s*=\s*([-+]?\d*\.\d+|\d+),\s*y\s*=\s*([-+]?\d*\.\d+|\d+)",
        "right_click": r"pyautogui\.rightClick\(x\s*=\s*([-+]?\d*\.\d+|\d+),\s*y\s*=\s*([-+]?\d*\.\d+|\d+)\)",
        "middle_click": r"pyautogui\.middleClick\(x\s*=\s*([-+]?\d*\.\d+|\d+),\s*y\s*=\s*([-+]?\d*\.\d+|\d+)\)",
        "scroll": r"pyautogui\.scroll\(\s*(-?\d*\.?\d+)\s*\)",
        "type": r"pyautogui\.write\(message\s*=\s*['\"](.*?)['\"]\s*\)",
        "key": r"pyautogui\.press\(\s*['\"](.*?)['\"]\s*\)",
    }
    
    # Precedence order: more specific with coordinates before generic
    for action, pattern in patterns.items():
        match = re.search(pattern, action_code)
        if match:
            args = {}
            if action in ["left_click", "right_click", "middle_click", "double_click", "mouse_move"]:
                x, y = int(float(match.group(1)) * 1000), int(float(match.group(2)) * 1000) # scale to 1000x1000
                args["coordinate"] = [x, y]
            elif action == "scroll":
                args["pixels"] = int(match.group(1))
            elif action == "type":
                args["text"] = match.group(1)
            elif action == "key":
                # Take arguments; match.group(1) yields something like: "'ctrl', 'c'"
                keys = [s.strip().strip("'\"") for s in match.group(1).split(",")]
                args["keys"] = keys
            tool_call_obj = {
                "name": action,
                "arguments": args
            }
            return f'<tool_call>\n{json.dumps(tool_call_obj, ensure_ascii=False)}\n</tool_call>'

    # Special-case for terminate/success/failure (if any present as code):
    if "wait" in action_code:
        args = {"time": 1}
        tool_call_obj = {
            "name": "wait",
            "arguments": args
        }
        return f'<tool_call>\n{json.dumps(tool_call_obj, ensure_ascii=False)}\n</tool_call>'
    if "terminate" in action_code:
        # For terminate action, look for explicit status
        if "success" in action_code:
            status = "success"
        elif "failure" in action_code:
            status = "failure"
        else:
            status = "success"
        args = {"status": status}
        tool_call_obj = {
            "name": "terminate",
            "arguments": args
        }
        return f'<tool_call>\n{json.dumps(tool_call_obj, ensure_ascii=False)}\n</tool_call>'

    # Fallback: return as free-form action string if not recognized
    args = {"raw_code": action_code}
    tool_call_obj = {
        "name": "unknown",
        "arguments": args
    }
    return f'<tool_call>\n{json.dumps(tool_call_obj, ensure_ascii=False)}\n</tool_call>'

def build_messages(sample):
    messages = [
        {
            "role": "system",
            "content": sample["system"]
        },
        {
            "role": "user",
            "content": sample["instruction"]
        }
    ]
    for message in sample["history_messages"]:
        messages.append(message)
    messages.append(sample["output"])
    return messages

def main():
    from tqdm import tqdm   
    trajectories = load_successful_trajectories(trajectory_dir)
    # breakpoint()
    
    training_data_samples = []
    test_data_samples = []

    import random

    # Shuffle the samples
    random.seed(42)
    random.shuffle(trajectories)

    # Split 9:1 (train:test)
    train_trajectories = trajectories[:int(0.9 * len(trajectories))]
    test_trajectories = trajectories[int(0.9 * len(trajectories)):]

    training_action_only_samples = []

    for trajectory in tqdm(train_trajectories):
        # breakpoint()
        action_only_data_samples_per_trajectory = construct_samples(trajectory)
        # breakpoint()
        # Step 1: build samples
        training_action_only_samples.extend(action_only_data_samples_per_trajectory)

    print(f"Constructed {len(training_action_only_samples)} action-only samples")

    training_data_samples = training_action_only_samples

    # from transformers import AutoProcessor

    # processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")

    # for sample in training_data_samples:
    #     messages = build_messages(sample)
    #     breakpoint()
    #     prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    # Save train set
    training_data_path = "datasets/opencua_dataset.json"
    with open(training_data_path, "w") as f:
        json.dump(training_data_samples, f, indent=4)

    print(f"Saved {len(training_data_samples)} training samples")
    
    for trajectory in tqdm(test_trajectories):
        data_samples_per_trajectory = construct_action_only_samples(trajectory)
        test_data_samples.extend(data_samples_per_trajectory)

    random.shuffle(test_data_samples)

    print(f"Saved {len(test_data_samples)} test samples")

    # Save test set in the current folder
    test_data_path = "datasets/opencua_dataset_test.json"
    with open(test_data_path, "w") as f:
        json.dump(test_data_samples, f, indent=4)

    
if __name__ == "__main__":
    main()