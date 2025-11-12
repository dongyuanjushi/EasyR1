import json

from transformers import AutoProcessor
from datetime import datetime
from PIL import Image
from uuid import uuid4
import re
import ast
import os
import math
from io import BytesIO
import subprocess
import glob
import signal

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

from typing import Optional, Any, List, Dict, Any, Union

import openai
import requests

import time

def call_llm_with_single_response(
    messages: List[Dict[str, str]], 
    llm_config: Dict[str, Any], 
    max_tokens: int = 1000, 
    temperature: float = 0.0, 
    top_p: float = 1.0, 
    response_format: Optional[Any] = None
) -> str:
    model = llm_config["model"]
    provider = llm_config["provider"]
    if provider == "bedrock":
        assert "endpoint" in llm_config, "endpoint is required for bedrock"
        litellm_client = openai.OpenAI(base_url=llm_config["endpoint"], api_key="token-123")
        while True:
            try:
                if response_format is not None:
                    completion = litellm_client.beta.chat.completions.parse(
                        model=f"{provider}/{model}",
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        response_format=response_format # TODO: add response_format back
                    )
                else:
                    completion = litellm_client.chat.completions.create(
                        model=f"{provider}/{model}",
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        extra_headers={
                            "anthropic-beta": "computer-use-2025-01-24"
                        }
                    )
                break

            except Exception as e:
                print(f"Error: {e}")
                time.sleep(10)
                continue
            
        return completion.choices[0].message.content
        
    elif provider == "vllm":
        assert "endpoint" in llm_config, "endpoint is required for vllm"
        vllm_client = openai.OpenAI(
            base_url=llm_config["endpoint"],
            api_key="token-abc123",
        )
        if response_format is not None:
            completion = vllm_client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                extra_body={"guided_json": response_format.model_json_schema()}
            )
        else:
            completion = vllm_client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                # top_p=top_p,
                # top_k=20,
                # top_p=0.8,
                # repetition_penalty=1.0,
                # presence_penalty=1.5
            )
        return completion.choices[0].message.content
    
    elif provider == "sglang":
        assert "endpoint" in llm_config, "endpoint is required for sglang"
        sglang_client = openai.OpenAI(
            base_url=llm_config["endpoint"],
            api_key="None",
        )
        if response_format is not None:
            completion = sglang_client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "foo",
                        # convert the pydantic model to json schema
                        "schema": response_format.model_json_schema(),
                    },
                }
            )
        else:
            completion = sglang_client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        return completion.choices[0].message.content

import base64
from typing import Union
from tqdm import tqdm

def bytes_literal_to_bytesio(bytes_literal_str):
    bytes_obj = ast.literal_eval(bytes_literal_str)

    if not isinstance(bytes_obj, bytes):
        raise ValueError("not a valid bytes literal")

def round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor

def ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor

def floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor

def smart_resize(height, width, factor=28, min_pixels=56 * 56, max_pixels=14 * 14 * 4 * 1280, max_long_side=8192):
    if height < 2 or width < 2:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(f"absolute aspect ratio must be smaller than 100, got {height} / {width}")

    if max(height, width) > max_long_side:
        beta = max(height, width) / max_long_side
        height, width = int(height / beta), int(width / beta)

    h_bar = round_by_factor(height, factor)
    w_bar = round_by_factor(width, factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar

def encode_screenshot(screenshot: Union[bytes, str]) -> str:
    if isinstance(screenshot, bytes):
        base_64_str = base64.b64encode(screenshot).decode("utf-8")
        return "data:image/jpeg;base64," + base_64_str
    elif isinstance(screenshot, str):
        bytes_obj = bytes_literal_to_bytesio(screenshot)
        base_64_str = base64.b64encode(bytes_obj).decode("utf-8")
        return "data:image/jpeg;base64," + base_64_str
    else:
        raise ValueError("type of screenshot is not supported, only bytes or str is supported")

def process_image(image_bytes):
    """
    Process an image for Qwen VL models (thinking variant).
    Uses a tighter resize cap consistent with the thinking DUN agent.
    """
    image = Image.open(BytesIO(image_bytes))
    width, height = image.size

    resized_height, resized_width = smart_resize(
        height=height,
        width=width,
        factor=32,
        min_pixels=32 * 32,
        max_pixels=16 * 16 * 4 * 12800,
        # max_pixels=1280*720,
    )

    image = image.resize((resized_width, resized_height))

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    processed_bytes = buffer.getvalue()

    return processed_bytes

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
                },
                "required": ["coordinate"]
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
                },
                "required": ["coordinate"]
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
                },
                "required": ["coordinate"]
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
                },
                "required": ["coordinate"]
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
                },
                "required": ["coordinate"]
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
                },
                "required": ["coordinate"]
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
                },
                "required": ["coordinate"]
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

SYSTEM_PROMPT_POLICY_MODEL = """You are a helpful assistant that can understand screenshots in the images and take actions in a computer environment to achieve the task. 
## Tools

You may call the tools defined below to assist with the given task.

Here are some tips for using the tools:
- Use a mouse and keyboard to interact with a computer, and take screenshots.",
- This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.",
- Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions. E.g. if you click on Firefox and a window doesn't open, try wait and taking another screenshot.",
- The screen's resolution is 1000x1000."
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
1) Action: a short imperative describing what to do in the UI.
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

def load_data(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data

def construct_input_for_vllm(sample):
    image_content = []

    for image_path in sample["images"]:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        image_bytes = process_image(image_bytes)
        image_content.append({
            "type": "image_url",
            "image_url": {
                "url": encode_screenshot(image_bytes)
            }
        })
    
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT_POLICY_MODEL
        },
        {
            "role": "user",
            "content": image_content + [{"type": "text", "text": f"{sample['instruction']}\n{sample['input'].replace('<image>', '')}"}]
        }
    ]
    
    return messages

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
            if predict_tool_call_args["status"] == gt_tool_call_args["status"]:
                return 1
            else:
                return 0
        else:
            return 0
    
def compute_score(output, ground_truth):
    format_score = compute_format_score(output)
    accuracy_score = compute_accuracy_score(output, ground_truth)
    if accuracy_score == 1:
        print(f"Prediction: {output}\nGround Truth: {ground_truth}")
    return {
        "format_score": format_score,
        "accuracy_score": accuracy_score,
        "total_score": 0.5 * format_score + 0.5 * accuracy_score
    }

def evaluate_sft(model_name, data, endpoint="http://localhost:30000/v1"):
    results = {
        "total_format_score": [],
        "total_accuracy_score": [],
    }
    
    for sample in tqdm(data[:200]):
        input_for_vllm = construct_input_for_vllm(sample)
        try:
            output = call_llm_with_single_response(
                messages=input_for_vllm,
                llm_config={
                    "model": model_name,
                    "provider": "vllm",
                    "endpoint": endpoint
                },
                temperature=0.7
            )
            score = compute_score(output, sample["output"])
        except Exception as e:
            score = {"format_score": 0, "accuracy_score": 0, "total_score": 0}

        _, gt_tool_call = parse_action_and_tool_call(sample["output"])
        action_type = gt_tool_call["name"]
        if action_type in ["left_click", "right_click", "middle_click", "double_click", "triple_click"]:
            action_key = "click"
            if action_key not in results:
                results[action_key] = {"format_score": [], "accuracy_score": []}
            results[action_key]["format_score"].append(score["format_score"])
            results[action_key]["accuracy_score"].append(score["accuracy_score"])

        elif action_type in ["scroll", "hscroll"]:
            action_key = "scroll"
            if action_key not in results:
                results[action_key] = {"format_score": [], "accuracy_score": []}
            results[action_key]["format_score"].append(score["format_score"])
            results[action_key]["accuracy_score"].append(score["accuracy_score"])
        elif action_type in ["type", "key"]:
            action_key = "type"
            if action_key not in results:
                results[action_key] = {"format_score": [], "accuracy_score": []}
            results[action_key]["format_score"].append(score["format_score"])
            results[action_key]["accuracy_score"].append(score["accuracy_score"])
        elif action_type in ["terminate"]:
            action_key = "terminate"
            if action_key not in results:
                results[action_key] = {"format_score": [], "accuracy_score": []}
            results[action_key]["format_score"].append(score["format_score"])
            results[action_key]["accuracy_score"].append(score["accuracy_score"])
        
        results["total_format_score"].append(score["format_score"])
        results["total_accuracy_score"].append(score["accuracy_score"])

    for action_key in results.keys():
        if action_key in ["click", "scroll", "type", "terminate"]:
            results[action_key]["avg_format_score"] = sum(results[action_key]["format_score"]) / len(results[action_key]["format_score"])
            results[action_key]["avg_accuracy_score"] = sum(results[action_key]["accuracy_score"]) / len(results[action_key]["accuracy_score"])

    results["avg_format_score"] = sum(results["total_format_score"]) / len(results["total_format_score"])
    results["avg_accuracy_score"] = sum(results["total_accuracy_score"]) / len(results["total_accuracy_score"])

    return results

def find_checkpoints(base_dir):
    # base_dir = "checkpoints/easy_r1/qwen3_vl_4b_opencua_grpo"
    # example_checkpoint = "checkpoints/easy_r1/qwen3_vl_4b_opencua_grpo/global_step_5/actor/huggingface"
    
    # Use the correct pattern for global_step_x/actor/huggingface, matching example_checkpoint above.
    checkpoint_pattern = os.path.join(base_dir, "global_step_*/actor/huggingface")

    checkpoint_dirs = glob.glob(checkpoint_pattern)
    # Filter out directories that do not contain any .safetensors file
    checkpoint_dirs = [
        d for d in checkpoint_dirs
        if any(
            fname.endswith(".safetensors")
            for fname in os.listdir(d)
            if os.path.isfile(os.path.join(d, fname))
        )
    ]
    # Sort by checkpoint number
    # checkpoint_dirs.sort(key=lambda x: int(os.path.basename(x).split("-")[-1]) if os.path.basename(x).split("-")[-1].isdigit() else 0)
    return checkpoint_dirs

def wait_for_server(endpoint, max_wait_time=600, check_interval=10):
    """Wait for the vLLM server to be ready."""
    start_time = time.time()
    # Try health endpoint first, then models endpoint
    health_endpoint = endpoint.replace("/v1", "/health")
    models_endpoint = f"{endpoint}/models"
    
    while time.time() - start_time < max_wait_time:
        try:
            # Try health endpoint
            try:
                response = requests.get(health_endpoint, timeout=2)
                if response.status_code == 200:
                    print(f"Server is ready at {endpoint}")
                    return True
            except:
                pass
            
            # Try models endpoint as fallback
            response = requests.get(models_endpoint, timeout=2)
            if response.status_code == 200:
                print(f"Server is ready at {endpoint}")
                return True
        except Exception as e:
            pass
        time.sleep(check_interval)
    
    raise TimeoutError(f"Server did not become ready within {max_wait_time} seconds")

def launch_vllm_server(checkpoint_path, port=30000, cuda_devices="0,1", tensor_parallel_size=2):
    """Launch a vLLM server for the given checkpoint."""
    cmd = [
        "vllm", "serve",
        checkpoint_path,
        "--tensor_parallel_size", str(tensor_parallel_size),
        "--host", "0.0.0.0",
        "--port", str(port),
        "--seed", "123"
    ]
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_devices
    
    print(f"Launching vLLM server for {checkpoint_path} on port {port}")
    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid  # Create new process group
    )
    
    return process

def terminate_server(process):
    """Terminate the vLLM server process."""
    if process is not None:
        try:
            # Kill the entire process group
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            process.wait(timeout=30)
            print("Server terminated successfully")
        except subprocess.TimeoutExpired:
            print("Server did not terminate gracefully, forcing kill")
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            process.wait()
        except Exception as e:
            print(f"Error terminating server: {e}")

def main():
    data = load_data("datasets/opencua_dataset_test.json")
    
    # Base directory containing checkpoints
    base_model_dir = "checkpoints/easy_r1/qwen3_vl_4b_opencua_grpo"
    
    # Find all checkpoints
    checkpoint_dirs = find_checkpoints(base_model_dir)
    
    if not checkpoint_dirs:
        print(f"No checkpoints found in {base_model_dir}")
        return
    
    print(f"Found {len(checkpoint_dirs)} checkpoints to evaluate")

    final_result = {}
    # Use the same port for all checkpoints since we terminate each server before starting the next
    port = 30000
    endpoint = f"http://localhost:{port}/v1"
    
    for idx, checkpoint_path in enumerate(checkpoint_dirs):
        print(f"\n{'='*80}")
        print(f"Evaluating checkpoint: {checkpoint_path} ({idx+1}/{len(checkpoint_dirs)})")
        print(f"{'='*80}")
        
        server_process = None
        try:
            # Launch server
            server_process = launch_vllm_server(
                checkpoint_path=checkpoint_path,
                port=port,
                cuda_devices="0,1",
                tensor_parallel_size=2
            )

            
            # Wait for server to be ready (give it more time for model loading)
            wait_for_server(endpoint, max_wait_time=600, check_interval=10)
            
            # Evaluate
            results = evaluate_sft(
                model_name=checkpoint_path,
                data=data,
                endpoint=endpoint
            )
            
            final_result[checkpoint_path] = results
            
            # Save intermediate results
            output_file = os.path.join(checkpoint_path, "sft_results.json")
            with open(output_file, "w") as f:
                json.dump(results, f, indent=4)
            print(f"Results saved to {output_file}")
            
        except Exception as e:
            print(f"Error evaluating {checkpoint_path}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Always terminate the server
            if server_process is not None:
                terminate_server(server_process)
                # Give some time for cleanup
                time.sleep(5)
    
    print(f"\n{'='*80}")
    print("Evaluation complete!")
    print(f"{'='*80}")
    print(final_result)
    
    # Save final results
    output_file = os.path.join(base_model_dir, "sft_results.json")
    with open(output_file, "w") as f:
        json.dump(final_result, f, indent=4)
    print(f"Final results saved to {output_file}")

if __name__ == "__main__":
    main()