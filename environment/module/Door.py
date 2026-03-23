from ..utils import api
from typing import Dict, Any, List


class Door:
    """
    Entity class representing doors with per-position settings.
    """

    PARAMS_DESCRIPTION = {
        "door": {
            "description": "The target door(s) for control operations. Supports individual doors, grouped doors, or all doors simultaneously.",
            "type": "str",
            "required": True,
            "valid_values": ["driver", "passenger", "rear_left", "rear_right", "front", "rear", "all"],
            "value_meanings": {
                "driver": "Driver side door (front left)",
                "passenger": "Front passenger side door (front right)",
                "rear_left": "Rear left passenger door",
                "rear_right": "Rear right passenger door",
                "front": "Both front doors (driver + passenger)",
                "rear": "Both rear doors (rear_left + rear_right)",
                "all": "All four doors simultaneously",
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
            "notes": "Chinese aliases are supported for convenience. If not specified, defaults to 'driver' door.",
        },
        "is_locked": {
            "description": "Lock state of the door that controls whether the door can be opened from inside or outside.",
            "type": "bool",
            "required": True,
            "valid_values": [True, False],
            "value_meanings": {
                True: "Lock the door - prevents the door from being opened until unlocked",
                False: "Unlock the door - allows the door to be opened from inside or outside",
            },
            "default": False,
            "notes": "Doors are locked by default for security. A door must be unlocked before it can be opened.",
        },
        "is_open": {
            "description": "Open/close state of the door.",
            "type": "bool",
            "required": True,
            "valid_values": [True, False],
            "value_meanings": {
                True: "Open the door - physically opens the door (requires door to be unlocked first)",
                False: "Close the door - physically closes the door",
            },
            "default": False,
            "notes": "IMPORTANT: Cannot open a door that is locked. You must first unlock the door (set is_locked to False) before opening it. Attempting to open a locked door will result in an error.",
        },
        "open_warning": {
            "description": "Door open warning alert that notifies when a door is left open.",
            "type": "bool",
            "required": True,
            "valid_values": [True, False],
            "value_meanings": {
                True: "Enable warning - system will alert when this door is open (useful when driving or leaving the vehicle)",
                False: "Disable warning - no alert when this door is open",
            },
            "default": False,
            "notes": "When enabled, the vehicle will provide visual and/or audible warnings if the door is open while the vehicle is in motion or when leaving the vehicle.",
        },
    }

    def __init__(self):
        """Initialize with default values for each door."""
        self._doors = {
            "driver": self._default_door_state(),
            "passenger": self._default_door_state(),
            "rear_left": self._default_door_state(),
            "rear_right": self._default_door_state(),
        }

    def _default_door_state(self) -> Dict[str, Any]:
        return {
            "is_locked": False,
            "is_open": False,
            "open_warning": False,
        }

    def _resolve_doors(self, door: str) -> List[str]:
        if not door:
            door = "driver"
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
        if door in mapping:
            return mapping[door]
        raise ValueError("door must be driver, passenger, rear_left, rear_right, front, rear, or all")

    def _set_for_doors(self, doors: List[str], key: str, value: Any) -> None:
        for door in doors:
            self._doors[door][key] = value

    # === API IMPLEMENTATION METHODS ===
    @api("door")
    def carcontrol_door_set_locked(self, door: str, locked: bool) -> Dict[str, Any]:
        """Set door lock state for a door or group."""
        try:
            doors = self._resolve_doors(door)
            self._set_for_doors(doors, "is_locked", bool(locked))
        except ValueError as exc:
            return {"success": False, "message": str(exc), "current_state": self.to_dict()}
        return {"success": True, "message": "Door lock state set successfully", "current_state": self.to_dict()}

    @api("door")
    def carcontrol_door_set_open(self, door: str, is_open: bool) -> Dict[str, Any]:
        """Set door open state for a door or group."""
        try:
            doors = self._resolve_doors(door)
            if is_open and any(self._doors[d]["is_locked"] for d in doors):
                raise ValueError("cannot open door when it is locked")
            self._set_for_doors(doors, "is_open", bool(is_open))
        except ValueError as exc:
            return {"success": False, "message": str(exc), "current_state": self.to_dict()}
        return {"success": True, "message": "Door open state set successfully", "current_state": self.to_dict()}

    @api("door")
    def carcontrol_door_set_open_warning(self, door: str, enabled: bool) -> Dict[str, Any]:
        """Enable or disable door open warning for a door or group."""
        try:
            doors = self._resolve_doors(door)
            self._set_for_doors(doors, "open_warning", bool(enabled))
        except ValueError as exc:
            return {"success": False, "message": str(exc), "current_state": self.to_dict()}
        return {"success": True, "message": "Door open warning set successfully", "current_state": self.to_dict()}

    def to_dict(self) -> Dict[str, Any]:
        """Convert door settings to a dictionary representation."""
        return {
            "driver": {
                "value": dict(self._doors["driver"]),
                "description": "Driver door settings",
                "type": type(self._doors["driver"]).__name__,
            },
            "passenger": {
                "value": dict(self._doors["passenger"]),
                "description": "Passenger door settings",
                "type": type(self._doors["passenger"]).__name__,
            },
            "rear_left": {
                "value": dict(self._doors["rear_left"]),
                "description": "Rear left door settings",
                "type": type(self._doors["rear_left"]).__name__,
            },
            "rear_right": {
                "value": dict(self._doors["rear_right"]),
                "description": "Rear right door settings",
                "type": type(self._doors["rear_right"]).__name__,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Door":
        """Create an instance from a dictionary representation."""
        instance = cls()
        instance._doors["driver"] = dict(data["driver"]["value"])
        instance._doors["passenger"] = dict(data["passenger"]["value"])
        instance._doors["rear_left"] = dict(data["rear_left"]["value"])
        instance._doors["rear_right"] = dict(data["rear_right"]["value"])
        return instance
