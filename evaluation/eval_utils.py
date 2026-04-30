import json
import os
import random
import re
import sys
import ast
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
from environment.utils import modules_dict

def extract_text(response, pattern):
    """
    Extract all content from response and organize by lines
    
    Args:
    response (str): Input text string
    
    Returns:
    str: Extracted API call content, each call on one line
    """
    api_calls = re.findall(pattern, response, re.DOTALL)
    
    api_calls = [call.strip() for call in api_calls]
    
    return '\n'.join(api_calls)

def add_modules(modules, module_num=0):
    random.seed(42)
    # Filter out existing keys and randomly select num modules
    available_modules = [key for key in modules_dict if key not in modules]
    
    select_modules = random.sample(available_modules, min(module_num, len(available_modules)))
    modules.extend(select_modules)

def _resolve_path(path_value: str) -> str:
    if os.path.isabs(path_value):
        return path_value
    return os.path.abspath(os.path.join(os.path.dirname(__file__), path_value))

def read_tasks(tasks_path="", module_num=0):
    """
    Reads task data from generated_task_queries.json.
    Each entry contains a generated query and reference tool list.

    Args:
        tasks_path (str): Path to the JSON file or a directory containing it
        module_num (int): Optional additional modules to add (kept for compatibility)

    Returns:
        list: List of task dictionaries with their data
    """
    tasks_path = _resolve_path(tasks_path)
    if os.path.isdir(tasks_path):
        tasks_path = os.path.join(tasks_path, "generated_task_queries.json")

    with open(tasks_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tasks = []
    for idx, item in enumerate(data):
        generated = item.get("generated", {})
        query = generated.get("query")
        if not query:
            continue
        modules = list(modules_dict.keys())
        if module_num > 0:
            add_modules(modules, module_num)
        tools = generated.get("tools", [])
        tasks.append({
            "id": idx,
            "query": query,
            "raw": item,
            "modules": modules,
            "tools": tools,
        })

    return tasks

def read_history(history_path: str) -> str:
    history_path = _resolve_path(history_path)
    with open(history_path, "r", encoding="utf-8") as f:
        return f.read()

def parse_tool_calls(code_str: str):
    if not code_str:
        return []
    calls = []
    try:
        tree = ast.parse(code_str)
    except SyntaxError:
        return calls
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            name = node.func.id
        else:
            continue
        kwargs = {}
        for kw in node.keywords:
            if kw.arg is None:
                continue
            try:
                kwargs[kw.arg] = ast.literal_eval(kw.value)
            except Exception:
                kwargs[kw.arg] = None
        calls.append({"name": name, "args": kwargs})
    return calls

def score_tool_calls(pred_calls, ref_calls):
    def to_key(item):
        name = item.get("name")
        args = item.get("args", {})
        return (name, json.dumps(args, sort_keys=True, ensure_ascii=False))

    pred_set = [to_key(c) for c in pred_calls]
    ref_set = [to_key(c) for c in ref_calls]

    tp = 0
    ref_used = set()
    for p in pred_set:
        if p in ref_set and p not in ref_used:
            tp += 1
            ref_used.add(p)
    fp = max(0, len(pred_set) - tp)
    fn = max(0, len(ref_set) - tp)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def parse_answer_to_tools(answer_list: list) -> list:
    """
    Convert a `new_answer` list into the structured `tools` format.

    Example:
    ["carcontrol_seat_set_headrest_height(seat=\"driver\", value=44)"]
    becomes:
    [{"name": "carcontrol_seat_set_headrest_height", "args": {"seat": "driver", "value": 44}}]
    """
    tools = []
    for answer_str in answer_list:
        match = re.match(r'(\w+)\((.*)\)', answer_str.strip())
        if not match:
            continue
        func_name = match.group(1)
        args_str = match.group(2)

        args = {}
        if args_str.strip():
            arg_pattern = r'(\w+)=("(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'|True|False|true|false|[\d.]+)'
            for arg_match in re.finditer(arg_pattern, args_str):
                key = arg_match.group(1)
                value_str = arg_match.group(2)
                if value_str.startswith('"') or value_str.startswith("'"):
                    value = value_str[1:-1]
                elif value_str.lower() == 'true':
                    value = True
                elif value_str.lower() == 'false':
                    value = False
                elif '.' in value_str:
                    value = float(value_str)
                else:
                    value = int(value_str)
                args[key] = value

        tools.append({"name": func_name, "args": args})
    return tools


def collect_values(obj, paths_map, path=""):
    """Collect important values from object"""
    if isinstance(obj, dict):
        # Handle special value objects
        if "type" in obj and "value" in obj:
            if obj["type"] in ["int", "str", "bool", "float"]:
                # Record value object
                paths_map[path] = obj["value"]
            
            # Handle cases where value is a list
            if isinstance(obj["value"], list):
                for index, item in enumerate(obj["value"]):
                    # Iterate through list elements for collection
                    new_path = f"{path}[{index}]"
                    collect_values(item, paths_map, new_path)
                
            if isinstance(obj["value"], dict):
                # Recursively process all values in dictionary
                for key, value in obj.items():
                    new_path = f"{path}.{key}" if path else key
                    collect_values(value, paths_map, new_path)
            
            return
        
        # Recursively process all values in dictionary
        for key, value in obj.items():
            new_path = f"{path}.{key}" if path else key
            collect_values(value, paths_map, new_path)
    
    # Handle lists
    elif isinstance(obj, list):
        # Record list length
        paths_map[path] = len(obj)
        
        # Recursively process list elements
        for i, item in enumerate(obj):
            collect_values(item, paths_map, f"{path}[{i}]")
    # It's a basic data type
    else:
        paths_map[path] = obj

def calculate_turn_result(world1, world2, world3, world4):
    """
    Calculate object change metrics based on comparison of expected and actual changes

    Args:
        world1: Original/reference object
        world2: Expected final object
        world3: Predicted initial object
        world4: Predicted final object

    Returns:
        dict: Contains difference list, TP/FP counts, change accuracy and F1 score
    """
    # Initialize result
    result = {
        "differences": [],
        "TP": 0,              # Should change and actually changed
        "FP": 0,              # Should not change but actually changed
        "negative_TP": 0,     # Should not change and actually didn't change
        "negative_FP": 0,     # Should change but actually didn't change
        "correctly_changed": 0, # Number of correctly changed items
    }

    # Collect value mappings of objects
    paths1 = {}  # Original reference object
    paths2 = {}  # Expected final object
    paths3 = {}  # Predicted initial object
    paths4 = {}  # Predicted final object

    # Collect values
    collect_values(world1, paths1)
    collect_values(world2, paths2)
    collect_values(world3, paths3)
    collect_values(world4, paths4)

    def values_equal(v1, v2):
        """Compare two values, case-insensitive for strings"""
        if isinstance(v1, str) and isinstance(v2, str):
            return v1.lower() == v2.lower()
        return v1 == v2

    # Step 1: Analyze world1 and world2 to determine which paths should change and which should not
    should_change_paths = []  # List of paths that should change
    should_not_change_paths = []  # List of paths that should not change

    # Get all paths in world1 and world2
    all_ref_paths = set(paths1.keys()) | set(paths2.keys())

    for path in all_ref_paths:
        has_path1 = path in paths1
        has_path2 = path in paths2

        # Case 1: Path exists in both world1 and world2
        if has_path1 and has_path2:
            val1 = paths1[path]
            val2 = paths2[path]

            # Check if it should change (use case-insensitive comparison for strings)
            if not values_equal(val1, val2):
                should_change_paths.append((path, val1, val2))
            else:
                should_not_change_paths.append(path)

        # Case 2: Path only exists in world1 - should be deleted
        elif has_path1 and not has_path2:
            should_change_paths.append((path, paths1[path], None))  # None means should be deleted

        # Case 3: Path only exists in world2 - should be added
        elif not has_path1 and has_path2:
            should_change_paths.append((path, None, paths2[path]))  # None means should be added

    # Step 2: Analyze world3 and world4 to check actual changes
    # For each path that should change, check if it actually changed
    for path_info in should_change_paths:
        path, expected_old, expected_new = path_info

        has_path3 = path in paths3
        has_path4 = path in paths4

        # Case 1: Should modify existing value
        if expected_old is not None and expected_new is not None:
            if has_path3 and has_path4:
                val3 = paths3[path]
                val4 = paths4[path]

                # Check if it actually changed (use case-insensitive comparison for strings)
                if not values_equal(val3, val4):
                    # Actually changed
                    result["TP"] += 1

                    # Check if the change is correct (case-insensitive exact match for all types)
                    if values_equal(expected_new, val4):
                        result["correctly_changed"] += 1
                    else:
                        result["differences"].append(f"{path}: Different value (should be {expected_new}, actual {val4})")
                else:
                    # Should change but didn't change
                    result["negative_FP"] += 1
                    result["differences"].append(f"{path}: Should change but didn't change")
            elif not has_path3 and has_path4:
                # Added, but should modify
                result["negative_FP"] += 1
                result["differences"].append(f"{path}: Should modify but was added")
            elif has_path3 and not has_path4:
                # Deleted, but should modify
                result["negative_FP"] += 1
                result["differences"].append(f"{path}: Should modify but was deleted")
            else:
                # Doesn't exist on both sides, cannot modify
                result["negative_FP"] += 1
                result["differences"].append(f"{path}: Doesn't exist, cannot modify")

        # Case 2: Should delete
        elif expected_old is not None and expected_new is None:
            if has_path3 and not has_path4:
                # Successfully deleted
                result["TP"] += 1
                result["correctly_changed"] += 1
            else:
                # Failed to delete
                result["negative_FP"] += 1
                if has_path3 and has_path4:
                    result["differences"].append(f"{path}: Should delete but not deleted")
                elif not has_path3 and not has_path4:
                    result["differences"].append(f"{path}: Doesn't exist originally, no need to delete")
                else:
                    result["differences"].append(f"{path}: Should delete but was added")

        # Case 3: Should add
        elif expected_old is None and expected_new is not None:
            if not has_path3 and has_path4:
                # Successfully added
                result["TP"] += 1

                val4 = paths4[path]
                # Check if added value is correct (use case-insensitive comparison for strings)
                if values_equal(expected_new, val4):
                    result["correctly_changed"] += 1
                else:
                    result["differences"].append(f"{path}: Added value incorrect (should be {expected_new}, actual {val4})")
            else:
                # Failed to add
                result["negative_FP"] += 1
                if has_path3 and has_path4:
                    result["differences"].append(f"{path}: Should add but already exists")
                elif has_path3 and not has_path4:
                    result["differences"].append(f"{path}: Should add but was deleted")
                else:
                    result["differences"].append(f"{path}: Should add but not added")

    # For each path that should not change, check if it actually changed
    for path in should_not_change_paths:
        has_path3 = path in paths3
        has_path4 = path in paths4

        if has_path3 and has_path4:
            val3 = paths3[path]
            val4 = paths4[path]

            if values_equal(val3, val4):
                # Correctly remained unchanged (use case-insensitive comparison for strings)
                result["negative_TP"] += 1
            else:
                # Should not change but changed
                result["FP"] += 1
                result["differences"].append(f"{path}: Should not change but changed ({val3} -> {val4})")
        elif has_path3 and not has_path4:
            # Should not change but was deleted
            result["FP"] += 1
            result["differences"].append(f"{path}: Should not change but was deleted")
        elif not has_path3 and has_path4:
            # Should not change but was added (doesn't exist in world3 but exists in world4)
            result["FP"] += 1
            result["differences"].append(f"{path}: Doesn't exist in world3 but was added in world4")
    
    # Check for unexpected additions in world4
    for path in paths4:
        if path not in all_ref_paths and path not in paths3:
            # Unexpected addition
            result["FP"] += 1 
            result["differences"].append(f"{path}: Unexpected addition")

    # Calculate change accuracy
    total_should_changed = len(should_change_paths)
    if total_should_changed > 0:
        result["change_accuracy"] = result["correctly_changed"] / total_should_changed
        result["skipped"] = False
    else:
        # If no attributes should change, the prediction is correct as long as
        # it also avoids incorrect modifications (FP == 0).
        result["change_accuracy"] = 1.0 if result["FP"] == 0 else 0.0
        result["skipped"] = False
    
    # Calculate F1 score - measure ability to change
    if total_should_changed > 0:
        # Precision: TP / (TP + FP)
        precision = result["TP"] / (result["TP"] + result["FP"]) if (result["TP"] + result["FP"]) > 0 else 0
        # Recall: TP / total_should_changed
        recall = result["TP"] / total_should_changed
        # F1 score
        result["f1_positive"] = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    else:
        # Precision: TP / (TP + FP)
        precision = result["TP"] / (result["TP"] + result["FP"]) if (result["TP"] + result["FP"]) > 0 else 1
        # Recall: TP / total_should_changed
        recall = 1
        # F1 score
        result["f1_positive"] = 2 * precision * recall / (precision + recall)
    
    # Calculate F1 score - measure ability to not change
    total_should_unchanged = len(should_not_change_paths)
    if total_should_unchanged > 0:
        # Precision: negative_TP / (negative_TP + negative_FP)
        precision = result["negative_TP"] / (result["negative_TP"] + result["negative_FP"]) if (result["negative_TP"] + result["negative_FP"]) > 0 else 0
        # Recall: negative_TP / total_should_unchanged
        recall = result["negative_TP"] / total_should_unchanged
        # F1 score
        result["f1_negative"] = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    else:
        # Precision: negative_TP / (negative_TP + negative_FP)
        precision = result["negative_TP"] / (result["negative_TP"] + result["negative_FP"]) if (result["negative_TP"] + result["negative_FP"]) > 0 else 1
        # Recall: negative_TP / total_should_changed
        recall = 1
        # F1 score
        result["f1_negative"] = 2 * precision * recall / (precision + recall)
    
    # ==================== NEW METRICS ====================
    
    # 1. acc_positive: Recall of field-level changes (TP / total_should_changed)
    #    Measures: What proportion of fields that should change were actually changed
    if total_should_changed > 0:
        result["acc_positive"] = result["TP"] / total_should_changed
    else:
        result["acc_positive"] = 1.0 if result["FP"] == 0 else 0.0
    
    # 2. precision_positive: Precision of field-level changes (TP / (TP + FP))
    #    Measures: What proportion of changed fields were supposed to be changed
    if (result["TP"] + result["FP"]) > 0:
        result["precision_positive"] = result["TP"] / (result["TP"] + result["FP"])
    else:
        result["precision_positive"] = 1.0 if total_should_changed == 0 else 0.0
    
    # 3. f1_change: F1 score based on correctly_changed (value-level F1)
    #    Measures: F1 score considering both precision and recall of correct value changes
    if total_should_changed > 0:
        # Precision: correctly_changed / (TP + FP) - proportion of changes that are correct
        precision_change = result["correctly_changed"] / (result["TP"] + result["FP"]) if (result["TP"] + result["FP"]) > 0 else 0
        # Recall: correctly_changed / total_should_changed - proportion of required changes done correctly
        recall_change = result["correctly_changed"] / total_should_changed
        # F1 score
        result["f1_change"] = 2 * precision_change * recall_change / (precision_change + recall_change) if (precision_change + recall_change) > 0 else 0
    else:
        # No changes required
        if result["FP"] == 0:
            result["f1_change"] = 1.0  # Perfect: nothing to change and nothing was changed incorrectly
        else:
            result["f1_change"] = 0.0  # Bad: changed something when nothing should change
    
    # 4. acc_negative: Accuracy of not changing fields that shouldn't change
    #    Measures: What proportion of fields that should NOT change remained unchanged
    if total_should_unchanged > 0:
        result["acc_negative"] = result["negative_TP"] / total_should_unchanged
    else:
        result["acc_negative"] = 1.0  # No fields should remain unchanged, so perfect by default
    
    # 5. precision_change: Precision of value-level correctness
    #    Measures: What proportion of all changes made were correct
    if (result["TP"] + result["FP"]) > 0:
        result["precision_change"] = result["correctly_changed"] / (result["TP"] + result["FP"])
    else:
        result["precision_change"] = 1.0 if total_should_changed == 0 else 0.0
    
    return result


