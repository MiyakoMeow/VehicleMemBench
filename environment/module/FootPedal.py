from ..utils import api
from typing import Dict, Any, List


class FootPedal:
    """
    Entity class representing foot pedals with per-seat switch states.
    """

    PARAMS_DESCRIPTION = {
        "position": {
            "description": "The target seat position for foot pedal control. Only front row seats have foot pedals.",
            "type": "str",
            "required": True,
            "valid_values": ["driver", "passenger", "all"],
            "value_meanings": {
                "driver": "Driver seat foot pedal (front left)",
                "passenger": "Front passenger seat foot pedal (front right)",
                "all": "Both front seat foot pedals simultaneously",
            },
            "aliases": {
                "主驾": "driver",
                "副驾": "passenger",
                "全部": "all",
            },
            "default": "driver",
            "notes": "Chinese aliases are supported for convenience. If not specified, defaults to 'driver' position. Rear seats do not have foot pedals.",
        },
        "switch": {
            "description": "Power state of the foot pedal/leg rest that controls whether it is extended or retracted.",
            "type": "bool",
            "required": True,
            "valid_values": [True, False],
            "value_meanings": {
                True: "Extend the foot pedal - raises the leg rest to support the passenger's legs for comfort",
                False: "Retract the foot pedal - lowers the leg rest back to the default position",
            },
            "default": False,
            "notes": "Foot pedals provide additional comfort for passengers during long trips. When extended, passengers can rest their legs in a more relaxed position.",
        },
    }

    def __init__(self):
        """Initialize with default values for each position."""
        self._pedals = {
            "driver": {"is_on": False},
            "passenger": {"is_on": False},
        }

    def _resolve_positions(self, position: str) -> List[str]:
        if not position:
            position = "driver"
        mapping = {
            "driver": ["driver"],
            "passenger": ["passenger"],
            "all": ["driver", "passenger"],
            "主驾": ["driver"],
            "副驾": ["passenger"],
            "全部": ["driver", "passenger"],
        }
        if position in mapping:
            return mapping[position]
        raise ValueError("position must be driver, passenger, or all")

    # === API IMPLEMENTATION METHODS ===
    @api("footPedal")
    def carcontrol_footPedal_set_switch(self, position: str, switch: bool) -> Dict[str, Any]:
        """Set foot pedal switch for a position or all."""
        try:
            positions = self._resolve_positions(position)
            for pos in positions:
                self._pedals[pos]["is_on"] = bool(switch)
        except ValueError as exc:
            return {"success": False, "message": str(exc), "current_state": self.to_dict()}
        return {
            "success": True,
            "message": "Foot pedal switch set successfully",
            "current_state": self.to_dict(),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert the foot pedal settings to a dictionary representation."""
        return {
            "driver": {
                "value": dict(self._pedals["driver"]),
                "description": "Driver foot pedal state",
                "type": type(self._pedals["driver"]).__name__,
            },
            "passenger": {
                "value": dict(self._pedals["passenger"]),
                "description": "Passenger foot pedal state",
                "type": type(self._pedals["passenger"]).__name__,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FootPedal":
        """Create an instance from a dictionary representation."""
        instance = cls()
        instance._pedals["driver"] = dict(data["driver"]["value"])
        instance._pedals["passenger"] = dict(data["passenger"]["value"])
        return instance
