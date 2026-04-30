"""
Model Evaluation for VehicleAgentBench
Unified evaluation pipeline for direct and memory-based settings.

Supported evaluation modes:
1. none: no memory context is provided to the model.
2. gold: preformatted gold context from each sample's `gold_memory` field is provided directly.
3. summary: recursive summarization where the LLM updates a summary each day,
   then the final summary is injected directly into the evaluation prompt.
4. key_value: tool-driven memory where the LLM maintains a KV store with
   `memory_add/remove`, then queries it with `memory_list/search` at evaluation time.

Evaluation flow:
1. For `none`, answer queries with no memory context.
2. For `gold`, answer queries with the sample's preformatted `gold_memory`.
3. For `summary` / `key_value`, load a history file and build memory day by day.
4. Answer queries from `related_to_vehicle_preference` using the selected context.
5. Compute a shared set of evaluation metrics across all modes.
"""

import json
import sys
import os
import inspect
import re
import time
import statistics
import traceback
import logging
import argparse
import random
from typing import Dict, List, Tuple, Optional, Any, get_type_hints
from collections import defaultdict
from datetime import datetime
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else []

    tqdm.write = print
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Add root dir
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from evaluation.agent_client import AgentClient
from evaluation.eval_utils import calculate_turn_result, parse_answer_to_tools, score_tool_calls
from environment.utils import save_json_file, modules_dict
from environment.vehicleworld import VehicleWorld

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)


def _detect_provider(model_name: str) -> str:
    """Detect API provider from model name for thinking mode configuration.
    
    Returns:
        Provider identifier: "deepseek", "spark", or "default"
    """
    model_lower = model_name.lower()
    if "deepseek" in model_lower:
        return "deepseek"
    if "spark-x" in model_lower or "sparkx" in model_lower:
        return "spark"
    return "default"


def create_chat_completion_with_retry(
    agent_client: AgentClient,
    *,
    max_retries: int = 3,
    retry_base_seconds: float = 3.0,
    context: str = "",
    **create_kwargs: Any
):
    """Call chat.completions.create with bounded retries.
    
    Supports thinking mode for multiple providers:
    - DeepSeek: Uses extra_body={"thinking": {"type": "enabled/disabled"}} and reasoning_effort
    - Spark-X: Uses extra_body={"thinking": {"type": "enabled/disabled"}}
    - Others: Uses extra_body={"enable_thinking": bool}
    """
    enable_thinking = getattr(agent_client, "enable_thinking", None)
    reasoning_effort = getattr(agent_client, "reasoning_effort", None)
    
    if enable_thinking is not None:
        extra_body = create_kwargs.get("extra_body")
        if not isinstance(extra_body, dict):
            extra_body = {}
        
        model_name = str(getattr(agent_client, "model", "") or "")
        provider = _detect_provider(model_name)
        
        if provider == "deepseek":
            # DeepSeek API format: {"thinking": {"type": "enabled/disabled"}}
            # Plus reasoning_effort parameter at top level
            extra_body["thinking"] = {
                "type": "enabled" if bool(enable_thinking) else "disabled"
            }
            # Set reasoning_effort for DeepSeek (high or max)
            # Per DeepSeek docs: low/medium/high → high, max → max
            if reasoning_effort is not None:
                # Map effort levels per DeepSeek documentation
                effort_mapping = {
                    "low": "high",
                    "medium": "high",
                    "high": "high",
                    "max": "max",
                }
                create_kwargs["reasoning_effort"] = effort_mapping.get(reasoning_effort, "high")
            elif "reasoning_effort" not in create_kwargs:
                # Default to "high" when thinking is enabled but effort not specified
                create_kwargs["reasoning_effort"] = "high"
        elif provider == "spark":
            # Spark-X API format: {"thinking": {"type": "enabled/disabled"}}
            extra_body["thinking"] = {
                "type": "enabled" if bool(enable_thinking) else "disabled"
            }
        else:
            # Default/generic format: {"enable_thinking": bool}
            extra_body["enable_thinking"] = bool(enable_thinking)
        
        create_kwargs["extra_body"] = extra_body

    for attempt in range(1, max_retries + 1):
        try:
            return agent_client.client.chat.completions.create(**create_kwargs)
        except Exception as e:
            if attempt == max_retries:
                raise
            context_suffix = f" ({context})" if context else ""
            logger.warning(
                f"Chat completion failed{context_suffix} (attempt {attempt}/{max_retries}): {e}. Retrying..."
            )
            time.sleep(retry_base_seconds * attempt)


def get_json_type(py_type):
    """Map a Python type to a JSON Schema type."""
    type_map = {
        int: "integer",
        float: "number",
        bool: "boolean",
        str: "string",
        list: "array",
        dict: "object",
    }
    return type_map.get(py_type, "string")


def get_functions_schema_for_module(module_name: str, vw: VehicleWorld = None):
    """Get the function schema for a specific module."""
    if vw is None:
        vw = VehicleWorld()

    functions = []
    module = getattr(vw, module_name, None)
    if not module:
        return functions

    for attr in dir(module):
        if not attr.startswith("carcontrol_"):
            continue

        func = getattr(module, attr, None)
        if not callable(func):
            continue

        doc = inspect.getdoc(func) or "No description"
        description = doc.split('\n')[0]

        sig = inspect.signature(func)
        type_hints = get_type_hints(func)

        properties = {}
        required = []

        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue

            param_type = type_hints.get(param_name, str)
            json_type = get_json_type(param_type)

            param_desc = param_name
            if param_name == "zone":
                param_desc = "Target zone: driver/passenger/rear_left/rear_right/front/rear/all"
            elif param_name in ["is_on", "enabled"]:
                param_desc = "Enable (true) or disable (false)"

            properties[param_name] = {
                "type": json_type,
                "description": param_desc,
            }

            if param.default == inspect.Parameter.empty:
                required.append(param_name)

        function_def = {
            "name": attr,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }
        functions.append(function_def)

    return functions


def get_list_module_tools_schema():
    """Return the schema for `list_module_tools`."""
    return {
        "name": "list_module_tools",
        "description": "List all available tools for a specific vehicle module. Call this first to discover what functions are available.",
        "parameters": {
            "type": "object",
            "properties": {
                "module_name": {
                    "type": "string",
                    "description": f"Module name. Available: {', '.join(modules_dict.keys())}",
                }
            },
            "required": ["module_name"],
        },
    }


def build_tool_env(vw: VehicleWorld):
    """Build the tool execution environment."""
    env = {"vw": vw}
    for key in modules_dict:
        module = getattr(vw, key, None)
        if not module:
            continue
        for attr in dir(module):
            if not attr.startswith("carcontrol_"):
                continue
            func = getattr(module, attr, None)
            if callable(func):
                env[attr] = func
    return env


# =============================================================================
# MemoryStore: key-value memory storage
# =============================================================================

class MemoryStore:
    """
    In-memory key-value storage with four tool operations:
    - memory_add(key, value): add or overwrite a memory entry
    - memory_remove(key): delete a memory entry
    - memory_search(key): search memory entries
    - memory_list(): list all stored keys
    """

    def __init__(self):
        self.store: Dict[str, str] = {}

    def memory_add(self, key: str, value: str) -> dict:
        """Add or overwrite a memory entry."""
        overwritten = key in self.store
        self.store[key] = value
        return {
            "success": True,
            "action": "updated" if overwritten else "added",
            "key": key,
            "value": value
        }

    def memory_remove(self, key: str) -> dict:
        """Delete a memory entry."""
        if key in self.store:
            del self.store[key]
            return {"success": True, "action": "removed", "key": key}
        return {"success": False, "error": f"Key '{key}' not found"}

    def memory_search(self, key: str) -> dict:
        """Search memory entries with exact and fuzzy matching."""
        # Exact match.
        if key in self.store:
            return {"success": True, "results": {key: self.store[key]}}

        # Fuzzy match: search key as a substring.
        results = {}
        key_lower = key.lower()
        for k, v in self.store.items():
            if key_lower in k.lower() or key_lower in v.lower():
                results[k] = v

        if results:
            return {"success": True, "results": results}
        return {"success": True, "results": {}, "message": "No matching entries found"}

    def memory_list(self) -> dict:
        """List all memory keys."""
        return {
            "success": True,
            "keys": list(self.store.keys()),
            "count": len(self.store)
        }

    def to_dict(self) -> dict:
        """Export all memory entries."""
        return dict(self.store)

    def to_text(self) -> str:
        """Format the store as text by returning the key list only."""
        if not self.store:
            return "No vehicle preferences recorded yet."
        return ', '.join(self.store.keys())


def get_memory_tools_schema(writable: bool = True, include_list: bool = True) -> List[dict]:
    """
    Generate the function schema for memory tools.

    Args:
        writable: If True, include add/remove for memory-building. Otherwise
            include only search/list for evaluation.
        include_list: If True, include memory_list. This is usually needed
            during evaluation but not during memory construction.
    """
    tools = []

    if writable:
        tools.append({
            "type": "function",
            "function": {
                "name": "memory_add",
                "description": "Add or update a memory entry. If the key already exists, the value will be overwritten.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "key": {
                            "type": "string",
                            "description": "A descriptive key for the memory entry (e.g., 'Gary_instrument_panel_color', 'Patricia_weekend_plan', 'Gary_workplace')"
                        },
                        "value": {
                            "type": "string",
                            "description": "The value to store (e.g., 'green', '44', 'enabled')"
                        }
                    },
                    "required": ["key", "value"]
                }
            }
        })
        tools.append({
            "type": "function",
            "function": {
                "name": "memory_remove",
                "description": "Remove a memory entry by key.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "key": {
                            "type": "string",
                            "description": "The key of the memory entry to remove"
                        }
                    },
                    "required": ["key"]
                }
            }
        })

    tools.append({
        "type": "function",
        "function": {
            "name": "memory_search",
            "description": "Search memory entries by key. Supports both exact and fuzzy matching.",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "The key or keyword to search for"
                    }
                },
                "required": ["key"]
            }
        }
    })
    if include_list:
        tools.append({
            "type": "function",
            "function": {
                "name": "memory_list",
                "description": "List all memory entry keys to see what preferences are stored.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        })

    return tools


# =============================================================================
# General utility: split history by day
# =============================================================================

def split_history_by_day(history_text: str) -> Dict[str, List[str]]:
    """Split history text into per-day conversation groups."""
    daily_conversations = defaultdict(list)
    pattern = r'\[(\d{4}-\d{2}-\d{2})\s+\d{2}:\d{2}\]'
    for line in history_text.strip().split('\n'):
        if not line.strip():
            continue
        match = re.match(pattern, line)
        if match:
            daily_conversations[match.group(1)].append(line)
    return dict(daily_conversations)


# =============================================================================
# Summary mode: tool-driven recursive summarization
# =============================================================================

def get_summary_memory_tools_schema() -> List[dict]:
    """
    Generate the summary-mode memory tool schema.

    Only one tool is exposed: `memory_update`, which rewrites the memory.
    """
    return [{
        "type": "function",
        "function": {
            "name": "memory_update",
            "description": "Update the memory with new vehicle-related preferences. Call this ONLY if you found NEW or CHANGED vehicle preferences in today's conversation. If no new vehicle information is found, do NOT call this tool.",
            "parameters": {
                "type": "object",
                "properties": {
                    "new_memory": {
                        "type": "string",
                        "description": "The complete updated memory content. This should include ALL existing preferences (from previous memory) plus any new/changed preferences from today's conversation. Format: bullet points organized by user name."
                    }
                },
                "required": ["new_memory"]
            }
        }
    }]


def summarize_day_with_previous_memory(
    agent_client: AgentClient,
    date: str,
    conversations: List[str],
    previous_memory: str = ""
) -> Tuple[str, bool, Optional[dict]]:
    """
    Tool-driven recursive summarization for daily memory updates.

    Returns:
        Tuple[str, bool, Optional[dict]]:
            (updated_memory, whether_a_tool_was_called, tool_call_log)
    """
    conversation_text = '\n'.join(conversations)
    tool_call_log = None

    system_prompt = """You are an intelligent assistant that maintains a CONCISE memory of user vehicle preferences from conversations.

**CRITICAL RULES:**
1. If today's conversation contains NEW vehicle-related information → Call memory_update with the complete updated memory
2. If today's conversation has NO new vehicle information → Do NOT call any tool, just respond that no update is needed
3. Keep the total memory under 2000 words
4. ONLY record vehicle-related preferences

**MUST capture (Priority 1 - Vehicle preferences):**
- In-car device settings: temperature, brightness, volume, seat position, ambient light color, navigation mode, HUD, air conditioning, massage, ventilation, instrument panel color, etc.
- Conditional preferences: e.g., "at night prefer white dashboard", "in industrial areas use inside circulation"
- User-specific preferences when there are conflicts between users
- Corrections or updates to previous settings

**MAY capture briefly (Priority 2 - Only if directly relevant to vehicle):**
- Frequently visited locations (for navigation): workplace, home address
- Physical conditions that affect vehicle settings: e.g., "back pain" → needs seat massage

**DO NOT capture:**
- General life events, plans, hobbies
- Work details unrelated to driving
- Personal relationships (unless affecting vehicle settings)

Output format for memory_update: Concise bullet points organized by user name.
Example:
**UserName**
- instrument_panel_color: green
- night_dashboard_preference: white (can't see gauges with dark colors at night)
- seat_massage: enabled when back hurts
"""

    if previous_memory and previous_memory != "No vehicle preferences recorded.":
        user_prompt = f"""**Current Memory (from previous days):**
{previous_memory}

**Today's Conversation ({date}):**
{conversation_text}

Analyze today's conversation:
- If there are NEW or CHANGED vehicle preferences → Call memory_update with the complete updated memory (existing + new)
- If there are NO new vehicle preferences → Do NOT call any tool"""
    else:
        user_prompt = f"""**Current Memory:** Empty (no preferences recorded yet)

**Today's Conversation ({date}):**
{conversation_text}

Analyze today's conversation:
- If there are any vehicle preferences mentioned → Call memory_update with the extracted preferences
- If there are NO vehicle preferences → Do NOT call any tool"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    available_tools = get_summary_memory_tools_schema()

    try:
        response = create_chat_completion_with_retry(
            agent_client,
            model=agent_client.model,
            messages=messages,
            temperature=agent_client.temperature,
            max_tokens=agent_client.max_tokens,
            tools=available_tools,
            tool_choice="auto",
            context=f"summarize_day_with_previous_memory date={date}"
        )

        message = response.choices[0].message
        tool_calls = message.tool_calls

        if tool_calls:
            # The LLM called memory_update.
            for tool_call in tool_calls:
                if tool_call.function.name == "memory_update":
                    try:
                        args = json.loads(tool_call.function.arguments)
                        new_memory = args.get("new_memory", "")

                        # Record the tool call for debugging and auditing.
                        tool_call_log = {
                            "name": "memory_update",
                            "args": args,
                            "result": {"success": True, "action": "updated"}
                        }

                        # Post-process the memory text to enforce a size limit.
                        MAX_MEMORY_LENGTH = 8192
                        if len(new_memory) > MAX_MEMORY_LENGTH:
                            truncate_pos = new_memory.rfind('\n-', 0, MAX_MEMORY_LENGTH)
                            if truncate_pos > MAX_MEMORY_LENGTH // 2:
                                new_memory = new_memory[:truncate_pos] + "\n[Memory truncated due to length limit]"
                            else:
                                new_memory = new_memory[:MAX_MEMORY_LENGTH] + "\n[Memory truncated due to length limit]"
                            logger.warning(f"Memory for {date} exceeded {MAX_MEMORY_LENGTH} chars, truncated.")

                        return new_memory, True, tool_call_log
                    except json.JSONDecodeError as e:
                        raw_args = tool_call.function.arguments
                        logger.error(f"Failed to parse tool arguments for {date}: {e}")
                        logger.error(f"Raw arguments (first 500 chars): {raw_args[:500] if raw_args else 'None'}")
                        # Try to extract new_memory directly as a fallback.
                        if raw_args and "new_memory" in raw_args:
                            try:
                                # Try a regex-based fallback extraction.
                                import re
                                match = re.search(r'"new_memory"\s*:\s*"(.*)"', raw_args, re.DOTALL)
                                if match:
                                    fallback_memory = match.group(1)
                                    # Unescape common escaped characters.
                                    fallback_memory = fallback_memory.replace('\\"', '"').replace('\\n', '\n')
                                    logger.warning(f"Using fallback regex extraction for {date}")
                                    return fallback_memory, True, {"name": "memory_update", "args": {"new_memory": fallback_memory}, "result": {"success": True, "action": "updated_fallback"}}
                            except Exception:
                                pass
                        return previous_memory, False, None

        # No tool call means no new memory was found.
        return previous_memory if previous_memory else "No vehicle preferences recorded.", False, None

    except Exception as e:
        logger.error(f"Error summarizing conversation for {date}: {e}")
        return previous_memory if previous_memory else "No vehicle preferences recorded.", False, None


def build_memory_recursive_summary(
    agent_client: AgentClient,
    daily_conversations: Dict[str, List[str]]
) -> Tuple[str, Dict[str, str], Dict[str, Optional[dict]]]:
    """
    Build a recursive summary-style memory store via tool calls.

    Returns:
        Tuple[str, Dict[str, str], Dict[str, Optional[dict]]]:
            (final_memory, daily_snapshots, daily_tool_logs)
    """
    daily_memory_snapshots = {}
    daily_tool_logs = {}
    accumulated_memory = ""
    sorted_dates = sorted(daily_conversations.keys())

    update_count = 0
    skip_count = 0

    for date in tqdm(sorted_dates, desc="Building recursive memory"):
        conversations = daily_conversations[date]
        accumulated_memory, was_updated, tool_log = summarize_day_with_previous_memory(
            agent_client, date, conversations, accumulated_memory
        )
        daily_memory_snapshots[date] = accumulated_memory
        daily_tool_logs[date] = tool_log

        if was_updated:
            update_count += 1
        else:
            skip_count += 1

    logger.info(f"Summary memory build complete: {update_count} updates, {skip_count} skipped")
    return accumulated_memory, daily_memory_snapshots, daily_tool_logs


# =============================================================================
# Key-value mode: memory construction via tool usage
# =============================================================================

def build_memory_kv_for_day(
    agent_client: AgentClient,
    date: str,
    conversations: List[str],
    memory_store: MemoryStore,
    reflect_num: int = 10
) -> List[dict]:
    """
    Build KV memory for a single day by letting the LLM use memory tools.

    Args:
        agent_client: LLM client.
        date: Date string.
        conversations: Conversations for the current day.
        memory_store: Memory store modified in place.
        reflect_num: Maximum number of tool-call rounds.

    Returns:
        List[dict]: Tool-call logs for the day, each with name, args, and result.
    """
    tool_call_logs = []  # Per-day tool-call log.
    conversation_text = '\n'.join(conversations)

    # Expose writable memory tools. memory_list is omitted because the prompt
    # already includes the current memory state.
    available_tools = get_memory_tools_schema(writable=True, include_list=False)

    # Inject current memory keys directly into the prompt.
    current_memory = memory_store.to_text()

    system_prompt = """You are an intelligent assistant that maintains a key-value memory store of user VEHICLE PREFERENCES from conversations.

**CRITICAL CONSTRAINTS:**
- ONLY store vehicle-related preferences and directly relevant context
- DO NOT store general life information (hobbies, plans, events, work details) unless they DIRECTLY affect vehicle settings
- Use concise values (under 50 characters per value)

Your task:
1. Review the current memory state provided below
2. Read today's conversation carefully
3. Extract and store ONLY vehicle-related information:

**MUST capture (Priority 1 - Vehicle preferences):**
- In-car device settings: temperature, brightness, volume, seat position, ambient light color, navigation mode, HUD, AC, massage, ventilation, instrument panel color
- Conditional preferences: e.g., "Gary_night_panel_color" = "white", "Patricia_industrial_area_circulation" = "inside"
- User-specific preferences when there are conflicts between users
- Corrections to previous settings (update the value)

**MAY capture briefly (Priority 2 - Only if directly relevant to vehicle):**
- Frequently visited locations (for navigation): "Justin_workplace" = "hospital"
- Physical conditions that affect vehicle settings: "Gary_back_condition" = "needs seat massage"

**DO NOT capture:**
- General life events, plans, hobbies
- Work details unrelated to driving
- Personal relationships (unless affecting vehicle settings)

4. Use memory_add() to store new entries or update changed ones
5. Use memory_remove() if information is explicitly revoked or outdated

If no vehicle-related information is mentioned today, do nothing."""

    user_prompt = f"""**Current Memory Keys:**
{current_memory}

**Today's Conversation ({date}):**
{conversation_text}

Please review the conversation and update the memory store with any VEHICLE-RELATED preferences found.
- ONLY store vehicle settings and preferences
- Ignore general life information, hobbies, plans, work details
- Use memory_search(key) to check existing values before updating"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # Tool execution mapping. Keep memory_list available just in case.
    tool_funcs = {
        "memory_add": memory_store.memory_add,
        "memory_remove": memory_store.memory_remove,
        "memory_search": memory_store.memory_search,
        "memory_list": memory_store.memory_list,
    }

    for _ in range(reflect_num):
        try:
            response = create_chat_completion_with_retry(
                agent_client,
                model=agent_client.model,
                messages=messages,
                temperature=agent_client.temperature,
                max_tokens=agent_client.max_tokens,
                tools=available_tools,
                tool_choice="auto",
                context=f"build_memory_kv_for_day date={date}"
            )
        except Exception as e:
            logger.error(f"API Error during KV memory build ({date}): {e}")
            break

        message = response.choices[0].message
        messages.append(message)

        tool_calls = message.tool_calls
        if not tool_calls:
            break

        for tool_call in tool_calls:
            func_name = tool_call.function.name
            try:
                args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                args = {}

            func = tool_funcs.get(func_name)
            if func:
                try:
                    # memory_list takes no arguments.
                    if func_name == "memory_list":
                        result = func()
                    else:
                        result = func(**args)
                except Exception as e:
                    result = {"success": False, "error": str(e)}
            else:
                result = {"success": False, "error": f"Unknown function: {func_name}"}

            # Record the tool call.
            tool_call_logs.append({
                "name": func_name,
                "args": args,
                "result": result
            })

            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": func_name,
                "content": json.dumps(result, ensure_ascii=False)
            })

    return tool_call_logs


def build_memory_key_value(
    agent_client: AgentClient,
    daily_conversations: Dict[str, List[str]],
    reflect_num: int = 20
) -> Tuple[MemoryStore, Dict[str, dict], Dict[str, List[dict]]]:
    """
    Build a key-value memory store via tool calls.

    Returns:
        Tuple[MemoryStore, Dict[str, dict], Dict[str, List[dict]]]:
            (final_memory_store, daily_snapshots, daily_tool_logs)
    """
    memory_store = MemoryStore()
    daily_snapshots = {}
    daily_tool_logs = {}  # Daily tool-call logs.
    sorted_dates = sorted(daily_conversations.keys())

    for date in tqdm(sorted_dates, desc="Building KV memory via tools"):
        conversations = daily_conversations[date]
        tool_logs = build_memory_kv_for_day(agent_client, date, conversations, memory_store, reflect_num)
        daily_snapshots[date] = memory_store.to_dict()
        daily_tool_logs[date] = tool_logs

    return memory_store, daily_snapshots, daily_tool_logs


# =============================================================================
# Evaluation stage: process_task logic shared across direct / memory modes
# =============================================================================

def _build_modules_info() -> str:
    return '\n'.join([f"- {k}: {v}" for k, v in modules_dict.items()])


def _handle_list_module_tools_call(args, loaded_modules, available_tools):
    """Load vehicle tools for a requested module."""
    module_name = args.get("module_name", "")
    tqdm.write(f"Loading tools for module: {module_name}")
    if module_name not in loaded_modules:
        module_functions = get_functions_schema_for_module(module_name)
        if module_functions:
            for func_schema in module_functions:
                available_tools.append({"type": "function", "function": func_schema})
            loaded_modules.add(module_name)
            return {
                "success": True,
                "message": f"Loaded {len(module_functions)} tools from {module_name} module",
                "tools": [f["name"] for f in module_functions]
            }
        return {"success": False, "error": f"Module {module_name} not found"}
    return {"success": True, "message": f"Module {module_name} already loaded"}


def _execute_named_tool(func_name, args, func_map, no_arg_tools=None):
    """Execute a named tool from a function map."""
    func = func_map.get(func_name)
    if not func:
        return {"success": False, "error": f"Function {func_name} not found"}

    try:
        if no_arg_tools and func_name in no_arg_tools:
            return func()
        return func(**args)
    except Exception as e:
        return {"success": False, "error": str(e)}


def _run_vehicle_task_evaluation(
    task,
    task_id,
    agent_client,
    reflect_num,
    system_instruction,
    request_context,
    initial_tools,
    memory_funcs=None,
):
    """Run the shared tool-calling evaluation loop."""
    try:
        query = task["query"]
        reasoning_type = task.get("reasoning_type", "unknown")

        vw_pred = VehicleWorld()
        local_vars = build_tool_env(vw_pred)
        vw_ref = VehicleWorld()

        available_tools = list(initial_tools)
        loaded_modules = set()
        messages = [{"role": "system", "content": system_instruction}]

        tqdm.write(f"Task {task_id}: {query[:60]}...")
        messages.append({"role": "user", "content": query})

        input_token_list = []
        output_token_list = []
        pred_calls = []
        all_tool_outputs = []
        no_arg_tools = {"memory_list"}

        for _ in range(reflect_num):
            try:
                response = create_chat_completion_with_retry(
                    agent_client,
                    model=agent_client.model,
                    messages=messages,
                    temperature=agent_client.temperature,
                    max_tokens=agent_client.max_tokens,
                    tools=available_tools,
                    tool_choice="auto",
                    context=request_context
                )
            except Exception as e:
                tqdm.write(f"API Error: {e}")
                break

            message = response.choices[0].message
            messages.append(message)

            usage = response.usage
            if usage:
                input_token_list.append(usage.prompt_tokens)
                output_token_list.append(usage.completion_tokens)

            tool_calls = message.tool_calls
            if not tool_calls:
                break

            for tool_call in tool_calls:
                func_name = tool_call.function.name
                try:
                    args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    args = {}

                if memory_funcs and func_name in memory_funcs:
                    pred_calls.append({"name": func_name, "args": args})
                    result = _execute_named_tool(func_name, args, memory_funcs, no_arg_tools=no_arg_tools)
                elif func_name == "list_module_tools":
                    result = _handle_list_module_tools_call(args, loaded_modules, available_tools)
                else:
                    pred_calls.append({"name": func_name, "args": args})
                    result = _execute_named_tool(func_name, args, local_vars)

                all_tool_outputs.append(str(result))
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": func_name,
                    "content": json.dumps(result, ensure_ascii=False)
                })

        return _build_task_result(
            task, query, reasoning_type, pred_calls, all_tool_outputs,
            vw_pred, vw_ref, messages, input_token_list, output_token_list, task_id
        )

    except Exception as e:
        stack_trace = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
        print(f"Task {task_id} error:\n{stack_trace}")
        return None

def process_task_direct(task, task_id, agent_client, reflect_num):
    """
    Direct function-calling evaluation without an explicit memory layer.

    This matches the historical direct-evaluation behavior.
    """
    history_text = task.get("history_text", "")
    modules_info = _build_modules_info()
    history_info = f"\n**Conversation History:**\n{history_text}\n" if history_text else ""

    system_instruction = f"""

{history_info}

You are an intelligent in-car AI assistant responsible for fulfilling user requests by calling the vehicle system API. You should analyze the current situation and perform the appropriate in-car operations to fulfill the user's request.

**Available Modules:**
{modules_info}

**Instructions:**
1. Use list_module_tools(module_name="xxx") to discover available functions
2. Call the specific functions you need
3. Check function descriptions for constraints and valid value ranges
4. Never ask the user for permission, clarification, or confirmation about vehicle
   operations. If memory or history provides relevant preferences, act on them immediately.
   You are an autonomous agent — execute, do not chat.
5. When the available information does not support setting a device to a specific value, avoid unnecessary parameter adjustments and perform only the minimal required action (e.g., just enabling the device or the corresponding attribute).
"""

    return _run_vehicle_task_evaluation(
        task=task,
        task_id=task_id,
        agent_client=agent_client,
        reflect_num=reflect_num,
        system_instruction=system_instruction,
        request_context=f"function_eval task={task_id}",
        initial_tools=[{"type": "function", "function": get_list_module_tools_schema()}],
    )

def process_task_with_memory(task, task_id, memory_text, agent_client, reflect_num):
    """
    Summary-mode evaluation where the final summary is injected into the prompt.

    The evaluation logic matches the direct evaluation path.
    """
    modules_info = _build_modules_info()
    memory_info = f"\n**User Preference Memory:**\n{memory_text}\n" if memory_text else ""

    system_instruction = f"""
{memory_info}

You are an intelligent in-car AI assistant responsible for fulfilling user requests by calling the vehicle system API. You should analyze the current situation and perform the appropriate in-car operations to fulfill the user's request.

**Available Modules:**
{modules_info}

**Instructions:**
1. Use list_module_tools(module_name="xxx") to discover available functions
2. Call the specific functions you need
3. Check function descriptions for constraints and valid value ranges
4. Never ask the user for permission, clarification, or confirmation about vehicle
   operations. If memory or history provides relevant preferences, act on them immediately.
   You are an autonomous agent — execute, do not chat.
5. When the available information does not support setting a device to a specific value, avoid unnecessary parameter adjustments and perform only the minimal required action (e.g., just enabling the device or the corresponding attribute).
6. Do not repeatedly query the same memory information or invoke the same vehicle tool in consecutive steps unless new evidence requires it.
"""

    return _run_vehicle_task_evaluation(
        task=task,
        task_id=task_id,
        agent_client=agent_client,
        reflect_num=reflect_num,
        system_instruction=system_instruction,
        request_context="evaluation_loop",
        initial_tools=[{"type": "function", "function": get_list_module_tools_schema()}],
    )


def process_task_with_kv_memory(task, task_id, memory_store, agent_client, reflect_num):
    """
    Key-value evaluation where the LLM queries memory via memory_list/search
    and then calls vehicle tools.
    """
    modules_info = _build_modules_info()
    system_instruction = f"""You are an intelligent in-car AI assistant responsible for fulfilling user requests by calling the vehicle system API. You should analyze the current situation and perform the appropriate in-car operations to fulfill the user's request.

You have access to a memory store containing user vehicle preferences. Use it to understand user preferences:
- memory_list(): List all stored preference keys
- memory_search(key): Search for specific preferences by keyword

**Available Vehicle Modules:**
{modules_info}

**Instructions:**
1. Use memory_list() and memory_search() to look up relevant user preferences
2. Use list_module_tools(module_name="xxx") to discover available vehicle functions
3. Call the specific vehicle functions based on user preferences and current request
4. Never ask the user for permission, clarification, or confirmation about vehicle
   operations. If memory or history provides relevant preferences, act on them immediately.
   You are an autonomous agent — execute, do not chat.
5. When the available information does not support setting a device to a specific value, avoid unnecessary parameter adjustments and perform only the minimal required action.
6. Do not repeatedly query the same memory information or invoke the same vehicle tool in consecutive steps unless new evidence requires it.
"""

    return _run_vehicle_task_evaluation(
        task=task,
        task_id=task_id,
        agent_client=agent_client,
        reflect_num=reflect_num,
        system_instruction=system_instruction,
        request_context="evaluation_loop",
        initial_tools=get_memory_tools_schema(writable=False) + [{"type": "function", "function": get_list_module_tools_schema()}],
        memory_funcs={
            "memory_search": memory_store.memory_search,
            "memory_list": memory_store.memory_list,
        },
    )


def _build_task_result(
    task, query, reasoning_type, pred_calls, all_tool_outputs,
    vw_pred, vw_ref, messages, input_token_list, output_token_list, task_id
):
    """Compute evaluation results shared by all memory modes."""
    ref_calls = task.get("tools", [])
    ref_env = build_tool_env(vw_ref)

    for ref_call in ref_calls:
        name = ref_call.get("name")
        args = ref_call.get("args", {})
        func = ref_env.get(name)
        if callable(func):
            try:
                func(**args)
            except Exception:
                pass

    initial_world_dict = VehicleWorld().to_dict()
    ref_world_dict = vw_ref.to_dict()
    pred_world_dict = vw_pred.to_dict()

    state_score = calculate_turn_result(
        initial_world_dict, ref_world_dict,
        initial_world_dict, pred_world_dict
    )
    tool_score = score_tool_calls(pred_calls, ref_calls)

    exact_match = (
        len(state_score.get("differences", [])) == 0 and
        state_score.get("FP", 0) == 0 and
        state_score.get("negative_FP", 0) == 0
    )

    skipped = state_score.get("skipped", False)
    if skipped:
        tqdm.write(f"[SKIP] Task {task_id}: Reference has no state change, query may have issues")

    last_msg = messages[-1]
    if hasattr(last_msg, 'content'):
        final_response = last_msg.content if last_msg.content else ""
    elif isinstance(last_msg, dict):
        final_response = last_msg.get('content', '')
    else:
        final_response = ""

    task_result = {
        "query": query,
        "reasoning_type": reasoning_type,
        "pred_calls": pred_calls,
        "ref_calls": ref_calls,
        "state_score": state_score,
        "exact_match": exact_match,
        "skipped": skipped,
        "tool_score": tool_score,
        "num_pred_calls": len(pred_calls),
        "num_ref_calls": len(ref_calls),
        "model_output": final_response,
        "system_return": all_tool_outputs,
        "input_token": statistics.mean(input_token_list) if input_token_list else 0,
        "output_token": sum(output_token_list) if output_token_list else 0,
        "task_id": task_id,
        "source_file": task.get("source_file", ""),
        "history_file": task.get("history_file", ""),
        "event_index": task.get("event_index", None),
    }

    tqdm.write(f"Task {task_id} done. Exact Match: {exact_match}, Tool F1: {tool_score['f1']:.2f}")
    return task_result


def _safe_mean(values):
    values = list(values)
    return statistics.mean(values) if values else 0.0


def _get_system_return_count(result):
    system_return = result.get("system_return", [])
    return len(system_return) if isinstance(system_return, list) else 0


def _build_metric(results, model, memory_type, extra_fields=None):
    """Build aggregate metrics shared by all evaluation modes."""
    valid_results = [r for r in results if not r.get("skipped", False)]
    skipped_results = [r for r in results if r.get("skipped", False)]

    metric = {
        "model": model,
        "memory_type": memory_type,
        "completed_tasks": len(results),
        "valid_tasks": len(valid_results),
        "skipped_tasks": len(skipped_results),
        "exact_match_rate": _safe_mean(1 if r.get("exact_match") else 0 for r in valid_results),
        "change_accuracy": _safe_mean(r.get("state_score", {}).get("change_accuracy", 0) for r in valid_results),
        "state_f1_positive": _safe_mean(r.get("state_score", {}).get("f1_positive", 0) for r in valid_results),
        "state_f1_negative": _safe_mean(r.get("state_score", {}).get("f1_negative", 0) for r in valid_results),
        "state_acc_positive": _safe_mean(r.get("state_score", {}).get("acc_positive", 0) for r in valid_results),
        "state_precision_positive": _safe_mean(r.get("state_score", {}).get("precision_positive", 0) for r in valid_results),
        "state_f1_change": _safe_mean(r.get("state_score", {}).get("f1_change", 0) for r in valid_results),
        "state_acc_negative": _safe_mean(r.get("state_score", {}).get("acc_negative", 0) for r in valid_results),
        "state_precision_change": _safe_mean(r.get("state_score", {}).get("precision_change", 0) for r in valid_results),
        "avg_pred_calls": _safe_mean(_get_system_return_count(r) for r in valid_results),
        "avg_output_token": _safe_mean(r.get("output_token", 0) for r in valid_results),
        "skipped_queries": [r.get("query", "")[:100] for r in skipped_results],
    }

    reasoning_types = set(r.get("reasoning_type", "unknown") for r in valid_results)
    metrics_by_type = {}
    for rtype in reasoning_types:
        type_results = [r for r in valid_results if r.get("reasoning_type") == rtype]
        if not type_results:
            continue
        metrics_by_type[rtype] = {
            "count": len(type_results),
            "exact_match_rate": _safe_mean(1 if r.get("exact_match") else 0 for r in type_results),
            "change_accuracy": _safe_mean(r.get("state_score", {}).get("change_accuracy", 0) for r in type_results),
            "state_f1_positive": _safe_mean(r.get("state_score", {}).get("f1_positive", 0) for r in type_results),
            "state_f1_negative": _safe_mean(r.get("state_score", {}).get("f1_negative", 0) for r in type_results),
            "state_acc_positive": _safe_mean(r.get("state_score", {}).get("acc_positive", 0) for r in type_results),
            "state_precision_positive": _safe_mean(r.get("state_score", {}).get("precision_positive", 0) for r in type_results),
            "state_f1_change": _safe_mean(r.get("state_score", {}).get("f1_change", 0) for r in type_results),
            "state_acc_negative": _safe_mean(r.get("state_score", {}).get("acc_negative", 0) for r in type_results),
            "state_precision_change": _safe_mean(r.get("state_score", {}).get("precision_change", 0) for r in type_results),
            "avg_pred_calls": _safe_mean(_get_system_return_count(r) for r in type_results),
            "avg_output_token": _safe_mean(r.get("output_token", 0) for r in type_results),
        }
    metric["by_reasoning_type"] = metrics_by_type

    if extra_fields:
        metric.update(extra_fields)

    return metric


def _print_metric_summary(metric):
    """Print aggregate metrics in a consistent format."""
    metrics_by_type = metric.get("by_reasoning_type", {})

    print("\n" + "=" * 60)
    print(f"Model Evaluation Results - {metric.get('model', 'unknown')} ({metric.get('memory_type', 'unknown')})")
    print("=" * 60)
    print(f"Tasks: {metric['valid_tasks']}/{metric['completed_tasks']} valid ({metric['skipped_tasks']} skipped)")

    print("\n--- Overall Metrics ---")
    print(f"Exact Match Rate:     {metric['exact_match_rate']:.2%}")

    print("\n  [Field-Level] (Whether the correct fields changed)")
    print(f"  Acc Positive:       {metric['state_acc_positive']:.2%}  (TP / fields that should change)")
    print(f"  Prec Positive:      {metric['state_precision_positive']:.2%}  (TP / changed fields)")
    print(f"  F1 Positive:        {metric['state_f1_positive']:.2%}")
    print("  ---")
    print(f"  Acc Negative:       {metric['state_acc_negative']:.2%}  (unchanged / fields that should stay unchanged)")
    print(f"  F1 Negative:        {metric['state_f1_negative']:.2%}")

    print("\n  [Value-Level] (Whether the correct value or trend was applied)")
    print(f"  Change Accuracy:    {metric['change_accuracy']:.2%}  (correct values / fields that should change)")
    print(f"  Prec Change:        {metric['state_precision_change']:.2%}  (correct values / changed fields)")
    print(f"  F1 Change:          {metric['state_f1_change']:.2%}")

    print("\n  [Efficiency]")
    print(f"  Avg Pred Calls:     {metric['avg_pred_calls']:.2f}")
    print(f"  Avg Output Token:   {metric['avg_output_token']:.2f}")

    print("\n--- Metrics by Reasoning Type ---")
    for rtype, rmetrics in sorted(metrics_by_type.items()):
        print(f"\n[{rtype}] (n={rmetrics['count']})")
        print(f"  Exact Match:        {rmetrics['exact_match_rate']:.2%}")
        print(f"  [Field] Acc/Prec/F1:  {rmetrics['state_acc_positive']:.2%} / {rmetrics['state_precision_positive']:.2%} / {rmetrics['state_f1_positive']:.2%}")
        print(f"  [Value] Acc/Prec/F1:  {rmetrics['change_accuracy']:.2%} / {rmetrics['state_precision_change']:.2%} / {rmetrics['state_f1_change']:.2%}")
        print(f"  [Negative] Acc/F1:    {rmetrics['state_acc_negative']:.2%} / {rmetrics['state_f1_negative']:.2%}")

    print("\n" + "=" * 60 + "\n")


def _save_final_outputs(output_subdir, metric, results, results_filename="results.json"):
    """Save aggregate artifacts shared by all modes."""
    save_json_file(metric, os.path.join(output_subdir, "metric.json"))
    save_json_file(results, os.path.join(output_subdir, results_filename))
    generate_report_txt(metric, os.path.join(output_subdir, "report.txt"))
    print(f"Results saved to {output_subdir}")


def _collect_event_chain_numbers(benchmark_dir, file_range):
    """Collect QA file numbers from a range string or by scanning the directory."""
    file_numbers = []
    if file_range:
        for part in file_range.split(","):
            part = part.strip()
            if "-" in part:
                start, end = part.split("-")
                file_numbers.extend(range(int(start), int(end) + 1))
            else:
                file_numbers.append(int(part))
    else:
        for filename in os.listdir(benchmark_dir):
            if filename.startswith("qa_") and filename.endswith(".json"):
                num = int(filename.replace("qa_", "").replace(".json", ""))
                file_numbers.append(num)
        file_numbers.sort()
    return file_numbers


def _collect_related_event_tasks(related_events, source_file, history_text_fn=None, extra_fields=None):
    """Convert related benchmark events into evaluation task dictionaries."""
    tasks = []
    for idx, event_item in enumerate(related_events):
        query = event_item.get("query", "")
        new_answer = event_item.get("new_answer", [])
        reasoning_type = event_item.get("reasoning_type", "unknown")

        if not query or not new_answer:
            continue

        task = {
            "query": query,
            "tools": parse_answer_to_tools(new_answer),
            "reasoning_type": reasoning_type,
            "source_file": source_file,
            "event_index": idx,
        }
        if history_text_fn is not None:
            task["history_text"] = history_text_fn(event_item)
        if extra_fields:
            task.update(extra_fields)
        tasks.append(task)
    return tasks


# =============================================================================
# Main entry points
# =============================================================================

def _evaluate_direct_mode(
    benchmark_dir,
    max_workers=1,
    sample_size=None,
    api_base="",
    api_key="",
    model="spark-x",
    reflect_num=20,
    prefix="model_eval",
    context_type="none",
    file_range=None,
    output_dir=None,
    resume_from_dir=None,
    enable_thinking=None,
    reasoning_effort=None,
    mode_label=None,
):
    """
    Direct function-calling evaluation.

    Args:
        context_type: "gold" for per-task gold context, or "none" for no context.
        mode_label: Optional public-facing mode name used in metrics/config.
        reasoning_effort: Reasoning effort level for thinking mode.
    """
    random.seed(42)

    if not os.path.isabs(benchmark_dir):
        benchmark_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), benchmark_dir))

    all_tasks = []
    file_numbers = _collect_event_chain_numbers(benchmark_dir, file_range)

    print(f"Processing QA files: {file_numbers}")

    if context_type not in ("gold", "none"):
        print(f"Unsupported context_type: {context_type}")
        return

    for num in file_numbers:
        file_path = os.path.join(benchmark_dir, f"qa_{num}.json")
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        all_tasks.extend(
            _collect_related_event_tasks(
                data.get("related_to_vehicle_preference", []),
                source_file=f"qa_{num}.json",
                history_text_fn=(
                    (lambda event_item: event_item.get("gold_memory", ""))
                    if context_type == "gold" else
                    (lambda event_item: "")
                ),
                extra_fields={"history_file": ""},
            )
        )

    print(f"Found {len(all_tasks)} tasks from {len(file_numbers)} QA files")

    if sample_size is None:
        sample_size = len(all_tasks)

    sample_tasks = all_tasks if sample_size >= len(all_tasks) else all_tasks[:sample_size]

    agent_client = AgentClient(api_base=api_base, api_key=api_key, model=model)
    agent_client.enable_thinking = enable_thinking
    agent_client.reasoning_effort = reasoning_effort

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "..", "log")
    if not os.path.isabs(output_dir):
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), output_dir))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if resume_from_dir:
        if not os.path.isabs(resume_from_dir):
            resume_from_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), resume_from_dir))
        output_subdir = resume_from_dir
        if not os.path.isdir(output_subdir):
            raise FileNotFoundError(f"resume_from_dir not found: {output_subdir}")
        print(f"Resume mode enabled. Output directory: {output_subdir}")
    else:
        output_subdir = os.path.join(output_dir, f"{prefix}_{model.replace('/', '_')}_{timestamp}")
        os.makedirs(output_subdir, exist_ok=True)

    config = {
        "timestamp": timestamp,
        "model": model,
        "memory_type": mode_label or context_type,
        "api_base": api_base,
        "context_type": context_type,
        "benchmark_dir": benchmark_dir,
        "file_range": file_range,
        "max_workers": max_workers,
        "reflect_num": reflect_num,
        "prefix": prefix,
        "sample_size": sample_size,
        "total_tasks": len(sample_tasks),
        "resume_from_dir": resume_from_dir,
        "enable_thinking": enable_thinking,
        "reasoning_effort": reasoning_effort,
    }
    config_name = "config.json"
    if resume_from_dir and os.path.exists(os.path.join(output_subdir, "config.json")):
        config_name = f"config_resume_{timestamp}.json"
    save_json_file(config, os.path.join(output_subdir, config_name))

    def build_result_keys(item):
        source_file = item.get("source_file", "")
        query = item.get("query", "")
        keys = {(source_file, "__query__", query)}
        event_index = item.get("event_index", None)
        if event_index is not None:
            keys.add((source_file, "__idx__", event_index))
        return keys

    results = []
    completed_task_keys = set()

    if resume_from_dir:
        existing_results_path = os.path.join(output_subdir, "results.json")
        if os.path.exists(existing_results_path):
            try:
                with open(existing_results_path, "r", encoding="utf-8") as f:
                    existing_results = json.load(f)
                if isinstance(existing_results, list):
                    results.extend(existing_results)
            except Exception as e:
                print(f"Warning: failed to load existing results.json: {e}")

        if not results:
            for filename in os.listdir(output_subdir):
                if not re.match(r"batch_\d+\.json$", filename):
                    continue
                file_path = os.path.join(output_subdir, filename)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        batch_data = json.load(f)
                    if isinstance(batch_data, list):
                        results.extend(batch_data)
                except Exception as e:
                    print(f"Warning: failed to load {filename}: {e}")

        for item in results:
            completed_task_keys.update(build_result_keys(item))
        print(f"Loaded {len(results)} existing results. Completed tasks: {len(completed_task_keys)}")

    pending_tasks = []
    for task in sample_tasks:
        task_keys = build_result_keys(task)
        if task_keys.isdisjoint(completed_task_keys):
            pending_tasks.append(task)
    print(f"Starting evaluation with {len(pending_tasks)} pending tasks ({len(sample_tasks)} total tasks)...")

    existing_batch_nums = []
    for filename in os.listdir(output_subdir):
        match = re.match(r"batch_(\d+)\.json$", filename)
        if match:
            existing_batch_nums.append(int(match.group(1)))
    next_batch_num = max(existing_batch_nums) + 1 if existing_batch_nums else 1

    batch_size = 500
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i in range(0, len(pending_tasks), batch_size):
            batch_num = next_batch_num + (i // batch_size)
            batch = pending_tasks[i:i + batch_size]
            batch_results = []

            futures = [
                executor.submit(
                    process_task_direct,
                    task,
                    idx + i,
                    agent_client,
                    reflect_num
                )
                for idx, task in enumerate(batch)
            ]

            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing batch {batch_num}"):
                res = future.result()
                if res:
                    results.append(res)
                    batch_results.append(res)

            save_json_file(batch_results, os.path.join(output_subdir, f"batch_{batch_num}.json"))

    if not results:
        return

    metric = _build_metric(
        results,
        model=model,
        memory_type=mode_label or context_type,
        extra_fields={"context_type": context_type},
    )
    _print_metric_summary(metric)
    _save_final_outputs(output_subdir, metric, results, results_filename="results.json")


def _evaluate_memory_mode(
    benchmark_dir: str,
    memory_type: str = "summary",
    sample_size: int = None,
    api_base: str = "",
    api_key: str = "",
    model: str = "gpt-4",
    reflect_num: int = 10,
    prefix: str = "model_eval",
    file_range: str = None,
    output_dir: str = None,
    save_memory: bool = True,
    max_workers: int = 6,
    resume_from_dir: str = None,
    enable_thinking: Optional[bool] = None,
    reasoning_effort: Optional[str] = None,
):
    """
    Internal helper for history-based memory evaluation.

    Args:
        benchmark_dir: Benchmark directory (`qa_data`).
        memory_type: Memory type ("summary" or "key_value").
        sample_size: Number of samples to keep per file.
        api_base: API base URL
        api_key: API key
        model: Model name.
        reflect_num: Maximum number of tool-call rounds.
        prefix: Output directory prefix.
        file_range: File-number range to process.
        output_dir: Output directory.
        save_memory: Whether to save built memory artifacts.
        max_workers: Maximum number of worker threads.
        resume_from_dir: Resume directory containing existing `results_*.json`.
        reasoning_effort: Reasoning effort level for thinking mode.
    """
    logger.info(f"Starting model evaluation with memory_type={memory_type}")

    if not os.path.isabs(benchmark_dir):
        benchmark_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), benchmark_dir))

    history_dir = os.path.join(os.path.dirname(benchmark_dir), "history")

    # Parse the requested file range.
    file_numbers = _collect_event_chain_numbers(benchmark_dir, file_range)

    # Keep only files whose event_chain and history files both exist.
    original_count = len(file_numbers)
    valid_file_numbers = []
    skipped_files = []
    for num in file_numbers:
        event_chain_path = os.path.join(benchmark_dir, f"qa_{num}.json")
        history_path = os.path.join(history_dir, f"history_{num}.txt")
        if os.path.exists(event_chain_path) and os.path.exists(history_path):
            valid_file_numbers.append(num)
        else:
            skipped_files.append(num)
    
    file_numbers = valid_file_numbers
    
    if skipped_files:
        logger.warning(f"Skipped {len(skipped_files)} files (not found): {skipped_files}")
    
    logger.info(f"Found {len(file_numbers)}/{original_count} valid files to process: {file_numbers}")

    agent_client = AgentClient(api_base=api_base, api_key=api_key, model=model)
    agent_client.enable_thinking = enable_thinking
    agent_client.reasoning_effort = reasoning_effort

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "..", "log")
    if not os.path.isabs(output_dir):
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), output_dir))

    if resume_from_dir:
        if not os.path.isabs(resume_from_dir):
            resume_from_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), resume_from_dir))
        output_subdir = resume_from_dir
        if not os.path.isdir(output_subdir):
            raise FileNotFoundError(f"resume_from_dir not found: {output_subdir}")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info(f"Resume mode enabled. Output directory: {output_subdir}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_subdir = os.path.join(output_dir, f"{prefix}_{memory_type}_{model.replace('/', '_')}_{timestamp}")
        os.makedirs(output_subdir, exist_ok=True)

    # Create the memory artifact subdirectory.
    memory_subdir = os.path.join(output_subdir, "memory")
    if save_memory:
        os.makedirs(memory_subdir, exist_ok=True)

    all_results = []
    results_lock = threading.Lock()  # Protect shared result aggregation.

    # Resume mode: load existing results_*.json and skip completed files.
    completed_files = set()
    if resume_from_dir:
        for filename in os.listdir(output_subdir):
            if not re.match(r"results_\d+\.json$", filename):
                continue
            match = re.match(r"results_(\d+)\.json$", filename)
            if not match:
                continue
            file_num = int(match.group(1))
            completed_files.add(file_num)
            file_path = os.path.join(output_subdir, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    all_results.extend(data)
                else:
                    logger.warning(f"Invalid format in {file_path}, expected list")
            except Exception as e:
                logger.warning(f"Failed to load existing result file {file_path}: {e}")

        logger.info(f"Loaded existing results from {len(completed_files)} files. Existing tasks: {len(all_results)}")

    pending_file_numbers = [n for n in file_numbers if n not in completed_files]
    if completed_files:
        logger.info(f"Skipping completed files: {sorted(completed_files)}")
    logger.info(f"Pending files to process: {pending_file_numbers}")

    def process_single_event_chain(num: int) -> List[dict]:
        """Process one event_chain file and return its evaluation results."""
        file_results = []

        logger.info(f"\n{'='*60}")
        logger.info(f"Processing qa_{num}")
        logger.info(f"{'='*60}")

        # Load QA file.
        event_chain_path = os.path.join(benchmark_dir, f"qa_{num}.json")
        if not os.path.exists(event_chain_path):
            logger.warning(f"File not found: {event_chain_path}")
            return file_results

        with open(event_chain_path, "r", encoding="utf-8") as f:
            event_chain = json.load(f)

        # Load history.
        history_path = os.path.join(history_dir, f"history_{num}.txt")
        if not os.path.exists(history_path):
            logger.warning(f"History not found: {history_path}")
            return file_results

        with open(history_path, "r", encoding="utf-8") as f:
            history_text = f.read()

        # Split history by day.
        daily_conversations = split_history_by_day(history_text)
        logger.info(f"[File {num}] Split history into {len(daily_conversations)} days")

        # ========================
        # Build memory
        # ========================
        logger.info(f"[File {num}] Building memory with type: {memory_type}")

        # Each worker thread needs its own AgentClient to avoid concurrency issues.
        thread_agent_client = AgentClient(api_base=api_base, api_key=api_key, model=model)
        thread_agent_client.enable_thinking = enable_thinking
        thread_agent_client.reasoning_effort = reasoning_effort

        memory_store = None  # Used by key_value mode.
        memory_text = ""     # Used by summary mode.
        daily_tool_logs = None  # Tool-call logs.

        if memory_type == "summary":
            memory_text, daily_snapshots, daily_tool_logs = build_memory_recursive_summary(thread_agent_client, daily_conversations)
            if save_memory:
                save_json_file(daily_snapshots, os.path.join(memory_subdir, f"memory_snapshots_{num}.json"))
                with open(os.path.join(memory_subdir, f"memory_{num}.txt"), "w", encoding="utf-8") as f:
                    f.write(memory_text)
                # Save tool-call logs.
                save_json_file(daily_tool_logs, os.path.join(memory_subdir, f"memory_tool_calls_{num}.json"))
            logger.info(f"[File {num}] Summary memory built. Length: {len(memory_text)} chars")

        elif memory_type == "key_value":
            # KV memory construction benefits from more reflection rounds than evaluation.
            kv_reflect_num = max(reflect_num, 20)
            if kv_reflect_num != reflect_num:
                logger.info(
                    "[File %d] KV memory reflect_num raised from %d to %d",
                    num, reflect_num, kv_reflect_num,
                )
            memory_store, daily_snapshots, daily_tool_logs = build_memory_key_value(
                thread_agent_client, daily_conversations, kv_reflect_num
            )
            if save_memory:
                save_json_file(daily_snapshots, os.path.join(memory_subdir, f"memory_snapshots_{num}.json"))
                save_json_file(memory_store.to_dict(), os.path.join(memory_subdir, f"memory_kv_{num}.json"))
                with open(os.path.join(memory_subdir, f"memory_{num}.txt"), "w", encoding="utf-8") as f:
                    f.write(memory_store.to_text())
                # Save tool-call logs.
                save_json_file(daily_tool_logs, os.path.join(memory_subdir, f"memory_tool_calls_{num}.json"))
            logger.info(f"[File {num}] KV memory built. Entries: {len(memory_store.store)}")

        else:
            logger.error(f"Unknown memory_type: {memory_type}")
            return file_results

        # ========================
        # Collect queries
        # ========================
        queries = _collect_related_event_tasks(
            event_chain.get("related_to_vehicle_preference", []),
            source_file=f"qa_{num}.json",
        )

        logger.info(f"[File {num}] Found {len(queries)} queries to evaluate")

        if sample_size is not None and sample_size < len(queries):
            queries = queries[:sample_size]
            logger.info(f"[File {num}] Sampled to {len(queries)} queries")

        # ========================
        # Evaluate
        # ========================
        for i, query_data in enumerate(tqdm(queries, desc=f"Evaluating queries (file {num})")):
            if memory_type == "summary":
                result = process_task_with_memory(
                    query_data, f"{num}_{i}", memory_text, thread_agent_client, reflect_num
                )
            elif memory_type == "key_value":
                result = process_task_with_kv_memory(
                    query_data, f"{num}_{i}", memory_store, thread_agent_client, reflect_num
                )
            else:
                continue

            if result:
                result["source_file"] = query_data["source_file"]
                result["event_index"] = query_data["event_index"]
                file_results.append(result)

        # Save results for this file.
        save_json_file(file_results, os.path.join(output_subdir, f"results_{num}.json"))
        logger.info(f"[File {num}] Completed. {len(file_results)} results saved.")

        return file_results

    # ========================
    # Process all event chains in parallel
    # ========================
    actual_workers = min(len(pending_file_numbers), max_workers)  # Actual number of worker threads.
    logger.info(f"Starting parallel processing with {actual_workers} workers for {len(pending_file_numbers)} pending files")

    if pending_file_numbers:
        with ThreadPoolExecutor(max_workers=actual_workers) as executor:
            # Submit all tasks.
            future_to_num = {executor.submit(process_single_event_chain, num): num for num in pending_file_numbers}

            # Collect results.
            for future in as_completed(future_to_num):
                num = future_to_num[future]
                try:
                    file_results = future.result()
                    with results_lock:
                        all_results.extend(file_results)
                    logger.info(f"[File {num}] Results collected. Total results so far: {len(all_results)}")
                except Exception as e:
                    logger.error(f"[File {num}] Failed with exception: {e}")
                    traceback.print_exc()

    if not all_results:
        logger.warning("No results to report")
        return

    metric = _build_metric(all_results, model=model, memory_type=memory_type)
    _print_metric_summary(metric)
    _save_final_outputs(output_subdir, metric, all_results, results_filename="results.json")

    config = {
        "timestamp": timestamp,
        "model": model,
        "memory_type": memory_type,
        "api_base": api_base,
        "benchmark_dir": benchmark_dir,
        "file_range": file_range,
        "reflect_num": reflect_num,
        "prefix": prefix,
        "sample_size": sample_size,
        "max_workers": max_workers,
        "resume_from_dir": resume_from_dir,
        "enable_thinking": enable_thinking,
        "reasoning_effort": reasoning_effort,
        "resumed_completed_files": sorted(completed_files),
        "resumed_pending_files": pending_file_numbers,
        "total_tasks": len(all_results),
    }
    save_json_file(config, os.path.join(output_subdir, "config.json"))

    logger.info(f"Results saved to {output_subdir}")


def model_evaluation(
    benchmark_dir: str,
    memory_type: str = "summary",
    sample_size: int = None,
    api_base: str = "",
    api_key: str = "",
    model: str = "gpt-4",
    reflect_num: int = 10,
    prefix: str = "model_eval",
    file_range: str = None,
    output_dir: str = None,
    save_memory: bool = True,
    max_workers: int = 6,
    resume_from_dir: str = None,
    enable_thinking: Optional[bool] = None,
    reasoning_effort: Optional[str] = None,
):
    """
    Unified public entry point for all evaluation modes.

    Supported modes:
    - none: no memory context
    - gold: use the sample's preformatted gold_memory as gold context
    - summary: build recursive summary memory from history
    - key_value: build tool-driven key-value memory from history
    
    Args:
        reasoning_effort: Reasoning effort level for thinking mode (e.g., "low", "medium", "high", "max").
                         Only effective when enable_thinking is True. Provider-specific behavior:
                         - DeepSeek: Maps to reasoning_effort parameter (high/max)
                         - Others: May be ignored or mapped accordingly
    """
    if memory_type == "none":
        return _evaluate_direct_mode(
            benchmark_dir=benchmark_dir,
            max_workers=max_workers,
            sample_size=sample_size,
            api_base=api_base,
            api_key=api_key,
            model=model,
            reflect_num=reflect_num,
            prefix=prefix,
            context_type="none",
            file_range=file_range,
            output_dir=output_dir,
            resume_from_dir=resume_from_dir,
            enable_thinking=enable_thinking,
            reasoning_effort=reasoning_effort,
            mode_label="none",
        )

    if memory_type == "gold":
        return _evaluate_direct_mode(
            benchmark_dir=benchmark_dir,
            max_workers=max_workers,
            sample_size=sample_size,
            api_base=api_base,
            api_key=api_key,
            model=model,
            reflect_num=reflect_num,
            prefix=prefix,
            context_type="gold",
            file_range=file_range,
            output_dir=output_dir,
            resume_from_dir=resume_from_dir,
            enable_thinking=enable_thinking,
            reasoning_effort=reasoning_effort,
            mode_label="gold",
        )

    if memory_type in ("summary", "key_value"):
        return _evaluate_memory_mode(
            benchmark_dir=benchmark_dir,
            memory_type=memory_type,
            sample_size=sample_size,
            api_base=api_base,
            api_key=api_key,
            model=model,
            reflect_num=reflect_num,
            prefix=prefix,
            file_range=file_range,
            output_dir=output_dir,
            save_memory=save_memory,
            max_workers=max_workers,
            resume_from_dir=resume_from_dir,
            enable_thinking=enable_thinking,
            reasoning_effort=reasoning_effort,
        )

    raise ValueError(f"Unsupported memory_type: {memory_type}")


def _fmt_pct(value):
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "N/A"

def _fmt_num(value):
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "N/A"

def generate_report_txt(metric, output_path):
    """Generate a formatted evaluation report and save it as a txt file."""
    lines = []
    model = metric.get("model", "unknown")
    memory_type = metric.get("memory_type", "unknown")
    lines.append("=" * 60)
    lines.append(f"Model Evaluation Results - {model} ({memory_type})")
    lines.append("=" * 60)
    valid = metric.get("valid_tasks", 0)
    completed = metric.get("completed_tasks", 0)
    skipped = metric.get("skipped_tasks", 0)
    lines.append(f"Tasks: {valid}/{completed} valid ({skipped} skipped)")
    lines.append("")
    lines.append("--- Overall Metrics ---")
    lines.append(f"Exact Match Rate:     {_fmt_pct(metric.get('exact_match_rate', 0))}")
    lines.append("")
    lines.append("  [Field-Level] (Whether the correct fields changed)")
    lines.append(f"  Acc Positive:       {_fmt_pct(metric.get('state_acc_positive', 0))}  (TP / fields that should change)")
    lines.append(f"  Prec Positive:      {_fmt_pct(metric.get('state_precision_positive', 0))}  (TP / changed fields)")
    lines.append(f"  F1 Positive:        {_fmt_pct(metric.get('state_f1_positive', 0))}")
    lines.append("  ---")
    lines.append(f"  Acc Negative:       {_fmt_pct(metric.get('state_acc_negative', 0))}  (unchanged / fields that should stay unchanged)")
    lines.append(f"  F1 Negative:        {_fmt_pct(metric.get('state_f1_negative', 0))}")
    lines.append("")
    lines.append("  [Value-Level] (Whether the correct value was applied)")
    lines.append(f"  Change Accuracy:    {_fmt_pct(metric.get('change_accuracy', 0))}  (correct values / fields that should change)")
    lines.append(f"  Prec Change:        {_fmt_pct(metric.get('state_precision_change', 0))}  (correct values / changed fields)")
    lines.append(f"  F1 Change:          {_fmt_pct(metric.get('state_f1_change', 0))}")
    lines.append("")
    lines.append("  [Efficiency]")
    lines.append(f"  Avg Pred Calls:     {_fmt_num(metric.get('avg_pred_calls', 0))}")
    lines.append(f"  Avg Output Token:   {_fmt_num(metric.get('avg_output_token', 0))}")

    by_type = metric.get("by_reasoning_type", {})
    if by_type:
        lines.append("")
        lines.append("--- Metrics by Reasoning Type ---")
        for rtype in sorted(by_type.keys()):
            rm = by_type[rtype]
            lines.append("")
            lines.append(f"[{rtype}] (n={rm.get('count', 0)})")
            lines.append(f"  Exact Match:        {_fmt_pct(rm.get('exact_match_rate', 0))}")
            lines.append(
                f"  [Field] Acc/Prec/F1:  "
                f"{_fmt_pct(rm.get('state_acc_positive', 0))} / "
                f"{_fmt_pct(rm.get('state_precision_positive', 0))} / "
                f"{_fmt_pct(rm.get('state_f1_positive', 0))}"
            )
            lines.append(
                f"  [Value] Acc/Prec/F1:  "
                f"{_fmt_pct(rm.get('change_accuracy', 0))} / "
                f"{_fmt_pct(rm.get('state_precision_change', 0))} / "
                f"{_fmt_pct(rm.get('state_f1_change', 0))}"
            )
            lines.append(
                f"  [Negative] Acc/F1:    "
                f"{_fmt_pct(rm.get('state_acc_negative', 0))} / "
                f"{_fmt_pct(rm.get('state_f1_negative', 0))}"
            )

    report_text = "\n".join(lines) + "\n"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"Report saved to {output_path}")


if __name__ == "__main__":
    def str2bool(v):
        if isinstance(v, bool):
            return v
        v = str(v).strip().lower()
        if v in ("true", "1", "yes", "y", "on"):
            return True
        if v in ("false", "0", "no", "n", "off"):
            return False
        raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")

    parser = argparse.ArgumentParser(description="Model evaluation for VehicleAgentBench")
    parser.add_argument("--benchmark_dir", type=str, default="../benchmark/qa_data",
                        help="Path to qa_data directory")
    parser.add_argument("--memory_type", type=str, default="summary",
                        choices=["none", "gold", "summary", "key_value"],
                        help="Evaluation mode: none, gold, summary, or key_value")
    parser.add_argument("--sample_size", type=int, default=None,
                        help="Number of samples per file to evaluate")
    parser.add_argument("--api_base", type=str, default="",
                        help="API base URL")
    parser.add_argument("--api_key", type=str, default="",
                        help="API key")
    parser.add_argument("--model", type=str, default="gpt-4",
                        help="Model name")
    parser.add_argument("--reflect_num", type=int, default=10,
                        help="Maximum number of tool call rounds")
    parser.add_argument("--prefix", type=str, default="model_eval",
                        help="Output file prefix")
    parser.add_argument("--file_range", type=str, default=None,
                        help="Range of files to process, e.g., '1-8' or '1,2,3'")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory")
    parser.add_argument("--no_save_memory", action="store_true",
                        help="Do not save memory to file")
    parser.add_argument("--max_workers", type=int, default=6,
                        help="Maximum number of parallel workers for processing event chains")
    parser.add_argument("--resume_from_dir", type=str, default=None,
                        help="Resume from an existing output directory; skip completed results_*.json files")
    parser.add_argument("--enable_thinking", type=str2bool, default=None,
                        help="Optional thinking mode flag (true/false). If omitted, the field is not sent.")
    parser.add_argument("--reasoning_effort", type=str, default=None,
                        choices=["low", "medium", "high", "max"],
                        help="Reasoning effort level for thinking mode (DeepSeek: low/medium map to high, max stays as max). Default: high when omitted with thinking enabled")

    args = parser.parse_args()

    model_evaluation(
        benchmark_dir=args.benchmark_dir,
        memory_type=args.memory_type,
        sample_size=args.sample_size,
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.model,
        reflect_num=args.reflect_num,
        prefix=args.prefix,
        file_range=args.file_range,
        output_dir=args.output_dir,
        save_memory=not args.no_save_memory,
        max_workers=args.max_workers,
        resume_from_dir=args.resume_from_dir,
        enable_thinking=args.enable_thinking,
        reasoning_effort=args.reasoning_effort,
    )
