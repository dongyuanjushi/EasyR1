import re
import json
from typing import Any

def parse_action_and_tool_call(output):
    action = ""
    tool_call = None
    # INSERT_YOUR_CODE
    # This regex matches any content between "Action: " and "<tool_call>", non-greedy.
    pattern_action_between = re.compile(r'Action:\s*(.*?)\s*<tool_call>', re.DOTALL)
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
    x1, y1 = predict_coordinate
    x2, y2 = gt_coordinate
    if abs(x1 - x2) <= 3 and abs(y1 - y2) <= 3:
        return 1
    else:
        return 0
    
def check_pixels(predict_pixels, gt_pixels):
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
        if predict_tool_call_name in ["left_click", "right_click", "middle_click", "double_click", "triple_click"]:
            try:
                predict_coordinate = predict_tool_call_args["coordinate"]
            except Exception as e:
                return 0
            if check_coordinate(predict_coordinate, gt_tool_call_args["coordinate"]):
                return 1
            else:
                return 0
        elif predict_tool_call_name in ["scroll", "hscroll"]:
            try:
                predict_pixels = predict_tool_call_args["pixels"]
            except Exception as e:
                return 0
            if check_pixels(predict_pixels, gt_tool_call_args["pixels"]):
                return 1
            else:
                return 0
        elif predict_tool_call_name in ["type", "key"]:
            try:
                predict_text = predict_tool_call_args["text"]
            except Exception as e:
                return 0
            if predict_text == gt_tool_call_args["text"]:
                return 1
            else:
                return 0
        elif predict_tool_call_name in ["terminate"]:
            try:
                predict_status = predict_tool_call_args["status"]
            except Exception as e:
                return 0
            if predict_tool_call_args["status"] == gt_tool_call_args["status"]:
                return 1
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
