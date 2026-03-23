from ..utils import api
from typing import Dict, Any, List


class Wiper:
    """
    Entity class representing wipers with per-position settings.
    """

    PARAMS_DESCRIPTION = {
        "wiper": {
            "description": "Target wiper position(s) for control.",
            "type": "str",
            "required": True,
            "valid_values": ["front", "rear", "all"],
            "value_meanings": {
                "front": "Front windshield wiper only",
                "rear": "Rear window wiper only",
                "all": "Both front and rear wipers",
            },
            "aliases": {
                "前": "front",
                "后": "rear",
                "全部": "all",
            },
            "default": "front",
            "notes": "Chinese aliases are supported.",
        },
        "is_on": {
            "description": "Power state of the wiper.",
            "type": "bool",
            "required": True,
            "valid_values": [True, False],
            "value_meanings": {
                True: "Turn on the wiper",
                False: "Turn off the wiper",
            },
            "default": False,
            "notes": "Will be auto-enabled when setting speed.",
        },
        "speed": {
            "description": "Wiper speed/frequency level.",
            "type": "int",
            "required": True,
            "valid_range": {"min": 1, "max": 5},
            "value_meanings": {
                1: "Slowest speed - intermittent wiping for light rain",
                2: "Low speed",
                3: "Medium speed - normal rain",
                4: "High speed",
                5: "Fastest speed - heavy rain",
            },
            "default": 1,
            "notes": "System will auto turn on wiper if needed. Higher speeds for heavier rain.",
        },
    }

    def __init__(self):
        """Initialize with default values for each wiper."""
        self._wipers = {
            "front": self._default_wiper_state(),
            "rear": self._default_wiper_state(),
        }

    def _default_wiper_state(self) -> Dict[str, Any]:
        return {
            "is_on": False,
            "speed": 1,
        }

    def _resolve_wipers(self, wiper: str) -> List[str]:
        if not wiper:
            wiper = "front"
        mapping = {
            "front": ["front"],
            "rear": ["rear"],
            "all": ["front", "rear"],
            "前": ["front"],
            "后": ["rear"],
            "全部": ["front", "rear"],
        }
        if wiper in mapping:
            return mapping[wiper]
        raise ValueError("wiper must be front, rear, or all")

    def _validate_range(self, value: int, min_value: int, max_value: int, name: str) -> int:
        if not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
        if value < min_value or value > max_value:
            raise ValueError(f"{name} must be between {min_value} and {max_value}")
        return value

    def _set_for_wipers(self, wipers: List[str], key: str, value: Any) -> None:
        for wiper in wipers:
            self._wipers[wiper][key] = value

    # === API IMPLEMENTATION METHODS ===
    @api("wiper")
    def carcontrol_wiper_set_open(self, wiper: str, is_on: bool) -> Dict[str, Any]:
        """Set wiper open state for a position or group."""
        try:
            wipers = self._resolve_wipers(wiper)
            self._set_for_wipers(wipers, "is_on", bool(is_on))
        except ValueError as exc:
            return {"success": False, "message": str(exc), "current_state": self.to_dict()}
        return {"success": True, "message": "Wiper open state set successfully", "current_state": self.to_dict()}

    @api("wiper")
    def carcontrol_wiper_set_speed(self, wiper: str, speed: int) -> Dict[str, Any]:
        """Set wiper speed (1-5). Auto turn on wiper if needed."""
        try:
            speed = self._validate_range(speed, 1, 5, "speed")
            wipers = self._resolve_wipers(wiper)
            # Auto turn on wiper when setting speed
            self._set_for_wipers(wipers, "is_on", True)
            self._set_for_wipers(wipers, "speed", speed)
        except ValueError as exc:
            return {"success": False, "message": str(exc), "current_state": self.to_dict()}
        return {"success": True, "message": "Wiper speed set successfully", "current_state": self.to_dict()}

    def to_dict(self) -> Dict[str, Any]:
        """Convert wiper settings to a dictionary representation."""
        return {
            "front": {
                "value": dict(self._wipers["front"]),
                "description": "Front wiper settings",
                "type": type(self._wipers["front"]).__name__,
            },
            "rear": {
                "value": dict(self._wipers["rear"]),
                "description": "Rear wiper settings",
                "type": type(self._wipers["rear"]).__name__,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Wiper":
        """Create an instance from a dictionary representation."""
        instance = cls()
        instance._wipers["front"] = dict(data["front"]["value"])
        instance._wipers["rear"] = dict(data["rear"]["value"])
        return instance
