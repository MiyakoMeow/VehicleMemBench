from ..utils import api
from typing import Dict, Any, List


class Window:
    """
    Entity class representing windows with per-position settings.
    """

    PARAMS_DESCRIPTION = {
        "window": {
            "description": "Target window position(s) for control.",
            "type": "str",
            "required": True,
            "valid_values": ["driver", "passenger", "rear_left", "rear_right", "front", "rear", "all"],
            "value_meanings": {
                "driver": "Driver side window only",
                "passenger": "Front passenger side window only",
                "rear_left": "Rear left passenger window only",
                "rear_right": "Rear right passenger window only",
                "front": "Both front windows (driver and passenger)",
                "rear": "Both rear windows (rear_left and rear_right)",
                "all": "All four windows",
            },
            "aliases": {
                "主驾": "driver",
                "副驾": "passenger",
                "左后": "rear_left",
                "右后": "rear_right",
                "前排": "front",
                "后排": "rear",
                "全部": "all",
            },
            "default": "driver",
            "notes": "Chinese aliases are supported.",
        },
        "is_open": {
            "description": "Open/close state of the window.",
            "type": "bool",
            "required": True,
            "valid_values": [True, False],
            "value_meanings": {
                True: "Open the window",
                False: "Close the window",
            },
            "default": False,
            "notes": "Will be auto-set based on open_degree: True when degree > 0, False when degree = 0.",
        },
        "open_degree": {
            "description": "Opening degree/percentage of the window.",
            "type": "int",
            "required": True,
            "valid_range": {"min": 0, "max": 100},
            "value_meanings": {
                0: "Fully closed (will auto set is_open to False)",
                50: "Half open",
                100: "Fully open",
            },
            "default": 0,
            "notes": "Setting degree > 0 will auto open window, setting degree = 0 will auto close window.",
        },
        "child_lock": {
            "description": "Child safety lock to prevent rear window operation from inside the vehicle.",
            "type": "bool",
            "required": True,
            "valid_values": [True, False],
            "value_meanings": {
                True: "Enable child lock - rear passengers cannot operate windows",
                False: "Disable child lock - normal operation",
            },
            "default": False,
            "notes": "Typically used for rear windows to prevent children from opening them.",
        },
        "auto_close_on_lock": {
            "description": "Automatically close window when vehicle is locked.",
            "type": "bool",
            "required": True,
            "valid_values": [True, False],
            "value_meanings": {
                True: "Enable auto close - window closes when car is locked",
                False: "Disable auto close",
            },
            "default": False,
            "notes": "Useful for security and weather protection when leaving the vehicle.",
        },
    }

    def __init__(self):
        """Initialize with default values for each window."""
        self._windows = {
            "driver": self._default_window_state(),
            "passenger": self._default_window_state(),
            "rear_left": self._default_window_state(),
            "rear_right": self._default_window_state(),
        }

    def _default_window_state(self) -> Dict[str, Any]:
        return {
            "is_open": False,
            "open_degree": 0,
            "child_lock": False,
            "auto_close_on_lock": False,
        }

    def _resolve_windows(self, window: str) -> List[str]:
        if not window:
            window = "driver"
        mapping = {
            "driver": ["driver"],
            "passenger": ["passenger"],
            "rear_left": ["rear_left"],
            "rear_right": ["rear_right"],
            "front": ["driver", "passenger"],
            "rear": ["rear_left", "rear_right"],
            "all": ["driver", "passenger", "rear_left", "rear_right"],
            "主驾": ["driver"],
            "副驾": ["passenger"],
            "左后": ["rear_left"],
            "右后": ["rear_right"],
            "前排": ["driver", "passenger"],
            "后排": ["rear_left", "rear_right"],
            "全部": ["driver", "passenger", "rear_left", "rear_right"],
        }
        if window in mapping:
            return mapping[window]
        raise ValueError("window must be driver, passenger, rear_left, rear_right, front, rear, or all")

    def _validate_range(self, value: int, min_value: int, max_value: int, name: str) -> int:
        if not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
        if value < min_value or value > max_value:
            raise ValueError(f"{name} must be between {min_value} and {max_value}")
        return value

    def _set_for_windows(self, windows: List[str], key: str, value: Any) -> None:
        for window in windows:
            self._windows[window][key] = value

    # === API IMPLEMENTATION METHODS ===
    @api("window")
    def carcontrol_window_set_open(self, window: str, is_open: bool) -> Dict[str, Any]:
        """Set window open state for a window or group."""
        try:
            windows = self._resolve_windows(window)
            self._set_for_windows(windows, "is_open", bool(is_open))
        except ValueError as exc:
            return {"success": False, "message": str(exc), "current_state": self.to_dict()}
        return {"success": True, "message": "Window open state set successfully", "current_state": self.to_dict()}

    @api("window")
    def carcontrol_window_set_open_degree(self, window: str, degree: int) -> Dict[str, Any]:
        """Set window open degree (0-100). Auto open when > 0, auto close when = 0."""
        try:
            degree = self._validate_range(degree, 0, 100, "open_degree")
            windows = self._resolve_windows(window)
            # Auto open when degree > 0, auto close when degree = 0
            if degree > 0:
                self._set_for_windows(windows, "is_open", True)
            self._set_for_windows(windows, "open_degree", degree)
            if degree == 0:
                self._set_for_windows(windows, "is_open", False)
        except ValueError as exc:
            return {"success": False, "message": str(exc), "current_state": self.to_dict()}
        return {"success": True, "message": "Window open degree set successfully", "current_state": self.to_dict()}

    @api("window")
    def carcontrol_window_set_child_lock(self, window: str, enabled: bool) -> Dict[str, Any]:
        """Enable or disable window child lock."""
        try:
            windows = self._resolve_windows(window)
            self._set_for_windows(windows, "child_lock", bool(enabled))
        except ValueError as exc:
            return {"success": False, "message": str(exc), "current_state": self.to_dict()}
        return {"success": True, "message": "Window child lock set successfully", "current_state": self.to_dict()}

    @api("window")
    def carcontrol_window_set_auto_close_on_lock(self, window: str, enabled: bool) -> Dict[str, Any]:
        """Enable or disable auto close on lock."""
        try:
            windows = self._resolve_windows(window)
            self._set_for_windows(windows, "auto_close_on_lock", bool(enabled))
        except ValueError as exc:
            return {"success": False, "message": str(exc), "current_state": self.to_dict()}
        return {"success": True, "message": "Window auto close on lock set successfully", "current_state": self.to_dict()}

    def to_dict(self) -> Dict[str, Any]:
        """Convert window settings to a dictionary representation."""
        return {
            "driver": {
                "value": dict(self._windows["driver"]),
                "description": "Driver window settings",
                "type": type(self._windows["driver"]).__name__,
            },
            "passenger": {
                "value": dict(self._windows["passenger"]),
                "description": "Passenger window settings",
                "type": type(self._windows["passenger"]).__name__,
            },
            "rear_left": {
                "value": dict(self._windows["rear_left"]),
                "description": "Rear left window settings",
                "type": type(self._windows["rear_left"]).__name__,
            },
            "rear_right": {
                "value": dict(self._windows["rear_right"]),
                "description": "Rear right window settings",
                "type": type(self._windows["rear_right"]).__name__,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Window":
        """Create an instance from a dictionary representation."""
        instance = cls()
        instance._windows["driver"] = dict(data["driver"]["value"])
        instance._windows["passenger"] = dict(data["passenger"]["value"])
        instance._windows["rear_left"] = dict(data["rear_left"]["value"])
        instance._windows["rear_right"] = dict(data["rear_right"]["value"])
        return instance
