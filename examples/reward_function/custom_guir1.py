import re
import json
from typing import Any

def parse_action_and_tool_call(output):
    action = ""
    tool_call = None
    # INSERT_YOUR_CODE
    # This regex matches any content between "Action: " and "<tool_call>", non-greedy.
    pattern_action_between = re.compile(r'<action>\s*(.*?)\s*</action>', re.DOTALL)
    action_match = pattern_action_between.search(output)
    if action_match:
        action = action_match.group(1).strip()
    # Use regex to extract the content between <tool_call> and </tool_call>
    tool_call_match = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', output, re.DOTALL)
    if tool_call_match:
        tool_call_content = tool_call_match.group(1).strip()
        try:
            tool_call = json.loads(tool_call_content)
        except Exception as e:
            tool_call = None
    
    return action, tool_call

def compute_format_score(output):
    action, tool_call = parse_action_and_tool_call(output)
    # breakpoint()
    if len(action) > 0 and tool_call is not None:
        return 1
    else:
        return 0
    
def check_coordinate(predict_coordinate, gt_coordinate):
    if not isinstance(predict_coordinate, list) or not isinstance(gt_coordinate, list):
        return 0
    if len(predict_coordinate) != 2 or len(gt_coordinate) != 2:
        return 0

    x1, y1 = predict_coordinate
    x2, y2 = gt_coordinate
    if not isinstance(x1, int) or not isinstance(y1, int) or not isinstance(x2, int) or not isinstance(y2, int):
        return 0
    if abs(x1 - x2) <= 3 and abs(y1 - y2) <= 3:
        return 1
    else:
        return 0
    
def check_pixels(predict_pixels, gt_pixels):
    if not isinstance(predict_pixels, int) or not isinstance(gt_pixels, int):
        return 0
    if abs(predict_pixels - gt_pixels) <= 3:
        return 1
    else:
        return 0

def compute_accuracy_score(output, ground_truth):
    predict_action, predict_tool_call = parse_action_and_tool_call(output)
    gt_action, gt_tool_call = parse_action_and_tool_call(ground_truth)
    # breakpoint()
    try:
        predict_tool_call_name = predict_tool_call["name"]
        predict_tool_call_args = predict_tool_call["arguments"]
    except Exception as e:
        return 0
   
    gt_tool_call_name = gt_tool_call["name"]
    gt_tool_call_args = gt_tool_call["arguments"]
    if predict_tool_call_name != gt_tool_call_name:
        return 0

    else:
        if predict_tool_call_name in ["left_click", "right_click", "middle_click", "double_click", "triple_click", "mouse_move"]:
            if "coordinate" in predict_tool_call_args and "coordinate" in gt_tool_call_args:
                predict_coordinate = predict_tool_call_args["coordinate"]
                gt_coordinate = gt_tool_call_args["coordinate"]
                if check_coordinate(predict_coordinate, gt_coordinate):
                    return 1
                else:
                    return 0
            elif "coordinate" in predict_tool_call_args or "coordinate" in gt_tool_call_args:
                return 0
            else:
                return 1

        elif predict_tool_call_name in ["scroll", "hscroll"]:
            if "pixels" in predict_tool_call_args and "pixels" in gt_tool_call_args:
                predict_pixels = predict_tool_call_args["pixels"]
                gt_pixels = gt_tool_call_args["pixels"]
                if check_pixels(predict_pixels, gt_pixels):
                    return 1
                else:
                    return 0
            else:
                return 0
        elif predict_tool_call_name in ["type", "key"]:
            if "text" in predict_tool_call_args and "text" in gt_tool_call_args:
                predict_text = predict_tool_call_args["text"]
                gt_text = gt_tool_call_args["text"]
                if not isinstance(predict_text, str) or not isinstance(gt_text, str):
                    return 0
                if predict_text == gt_text:
                    return 1
                else:
                    return 0
            else:
                return 0
        elif predict_tool_call_name in ["terminate"]:
            if "status" in predict_tool_call_args and "status" in gt_tool_call_args:
                predict_status = predict_tool_call_args["status"]
                gt_status = gt_tool_call_args["status"]
                if not isinstance(predict_status, str) or not isinstance(gt_status, str):
                    return 0
                if predict_status == gt_status:
                    return 1
                else:
                    return 0
            else:
                return 0
        else:
            return 0

def compute_score(reward_input: dict[str, Any], format_weight: float = 0.9) -> dict[str, float]:
    print(reward_input)
    if not isinstance(reward_input, dict):
        raise ValueError("Please use `reward_type=sequential` for r1v reward function.")

    response = reward_input["response"]
    ground_truth = reward_input["ground_truth"]
    format_score = compute_format_score(response)
    accuracy_score = compute_accuracy_score(response, ground_truth)
    return {
        "overall": (1 - format_weight) * format_score + format_weight * accuracy_score,
        "format": format_score,
        "accuracy": accuracy_score,
    }
