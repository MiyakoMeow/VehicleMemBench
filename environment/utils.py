
import ast
import io
import json
from contextlib import redirect_stdout
from typing import Any, Dict, Iterable, Optional
from functools import wraps

# Create a global dictionary to store decorated methods, grouped by type
apis: Dict[str, list] = {}

modules_dict = {
    "HUD": "Head-up display related",
    "centerInformationDisplay": "Center information display related",
    "instrumentPanel": "Instrument panel related",
    "frontTrunk": "Front trunk related",
    "trunk": "Trunk related",
    "fuelPort": "Fuel port related",
    "rearviewMirror": "Rearview mirror related",
    "sunroof": "Sunroof related",
    "navigation": "Vehicle navigation system related",
    "seat": "Seat related",
    "radio": "Vehicle radio system related",
    "airConditioner": "Vehicle air conditioning system related",
    "footPedal": "Foot pedal related",
    "bluetooth": "Vehicle Bluetooth related",
    "video": "Vehicle video system related",
    "window": "Window related",
    "door": "Door related",
    "sunshade": "Sunshade related",
    "wiper": "Wiper related",
    "music": "Vehicle music system related",
    "overheadScreen": "Overhead screen related",
    "steeringWheel": "Steering wheel related",
    "light": "Light related",
}


def _diff_state(before: Any, after: Any) -> Any:
    if isinstance(before, dict) and isinstance(after, dict):
        if set(after.keys()) >= {"value", "description", "type"} and "value" in before:
            return after.get("value") if before.get("value") != after.get("value") else {}
        diff = {}
        for key, after_val in after.items():
            if key not in before:
                diff[key] = after_val.get("value") if isinstance(after_val, dict) and "value" in after_val else after_val
                continue
            before_val = before[key]
            sub = _diff_state(before_val, after_val)
            if sub not in ({}, [], None):
                diff[key] = sub
        return diff
    if isinstance(before, list) and isinstance(after, list):
        return after if before != after else []
    return after if before != after else {}

def api(module_name: str):
    def decorator(func):
        apis.setdefault(module_name, []).append(func.__name__)

        @wraps(func)
        def wrapper(*args, **kwargs):
            self_obj = args[0] if args else None
            before = self_obj.to_dict() if self_obj and hasattr(self_obj, "to_dict") else None
            result = func(*args, **kwargs)
            after = self_obj.to_dict() if self_obj and hasattr(self_obj, "to_dict") else None
            if isinstance(result, dict) and "current_state" in result and before is not None and after is not None:
                result["current_state"] = _diff_state(before, after)
            return result

        return wrapper
    return decorator


def get_api_content(modules: Optional[Iterable[str]] = None) -> Dict[str, list]:
    if modules is None:
        return {k: list(v) for k, v in apis.items()}
    return {name: list(apis.get(name, [])) for name in modules}


def capitalize_first(value: str) -> str:
    if not value:
        return value
    return value[0].upper() + value[1:]


def execute(code: str, local_vars: Optional[Dict[str, Any]] = None, global_vars: Optional[Dict[str, Any]] = None):
    if not code or not code.strip():
        return ""
    local_vars = local_vars or {}
    if global_vars is None:
        global_vars = local_vars
    outputs = []
    buffer = io.StringIO()
    try:
        tree = ast.parse(code)
        with redirect_stdout(buffer):
            for node in tree.body:
                if isinstance(node, ast.Expr):
                    value = eval(compile(ast.Expression(node.value), "<eval>", "eval"), global_vars, local_vars)
                    if value is not None:
                        outputs.append(value)
                else:
                    exec(compile(ast.Module([node], type_ignores=[]), "<exec>", "exec"), global_vars, local_vars)
    except Exception as exc:
        return {"success": False, "error": str(exc)}

    printed = buffer.getvalue().strip()
    if printed:
        outputs.append(printed)
    if not outputs:
        return {"success": True}
    return outputs[0] if len(outputs) == 1 else outputs


def save_json_file(data: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
