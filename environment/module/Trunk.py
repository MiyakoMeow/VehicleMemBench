from ..utils import api
from typing import Dict, Any


class Trunk:
    """
    Entity class representing the rear trunk.
    """

    PARAMS_DESCRIPTION = {
        "is_on": {
            "description": "Power state of the rear trunk electric system.",
            "type": "bool",
            "required": True,
            "valid_values": [True, False],
            "value_meanings": {
                True: "Turn on the trunk system - enables electric opening/closing",
                False: "Turn off the trunk system",
            },
            "default": False,
            "notes": "Will be auto-enabled when setting open_degree > 0, and auto-disabled when setting open_degree = 0.",
        },
        "open_degree": {
            "description": "Opening degree/percentage of the rear trunk lid.",
            "type": "int",
            "required": True,
            "valid_range": {"min": 0, "max": 100},
            "value_meanings": {
                0: "Fully closed (will auto turn off)",
                50: "Half open",
                100: "Fully open",
            },
            "default": 0,
            "notes": "Setting degree > 0 will auto turn on, setting degree = 0 will auto turn off.",
        },
    }

    def __init__(self):
        """Initialize with default values."""
        self._is_on = False
        self._open_degree = 0

    # === POWER STATE ===
    @property
    def is_on(self) -> bool:
        """Get rear trunk state."""
        return self._is_on

    @is_on.setter
    def is_on(self, value: bool):
        """Set rear trunk state."""
        self._is_on = bool(value)

    # === OPEN DEGREE ===
    @property
    def open_degree(self) -> int:
        """Get open degree (0-100)."""
        return self._open_degree

    @open_degree.setter
    def open_degree(self, value: int):
        """Set open degree (0-100)."""
        self._open_degree = self._validate_open_degree(value)

    def _validate_open_degree(self, value: int) -> int:
        if not isinstance(value, int):
            raise ValueError("open_degree must be an integer")
        if value < 0 or value > 100:
            raise ValueError("open_degree must be between 0 and 100")
        return value

    # === API IMPLEMENTATION METHODS ===
    @api("trunk")
    def carcontrol_trunk_switch(self, switch: bool) -> Dict[str, Any]:
        """Turn the rear trunk system on or off."""
        self.is_on = switch
        return {
            "success": True,
            "message": f"Trunk {'activated' if switch else 'deactivated'} successfully",
            "current_state": self.to_dict(),
        }

    @api("trunk")
    def carcontrol_trunk_set_open_degree(self, degree: int) -> Dict[str, Any]:
        """Set rear trunk open degree (0-100). Auto on when > 0, auto off when = 0."""
        try:
            # Auto turn on when opening, auto turn off when closing
            if degree > 0:
                self._is_on = True
            self.open_degree = degree
            if degree == 0:
                self._is_on = False
        except ValueError as exc:
            return {
                "success": False,
                "message": str(exc),
                "current_state": self.to_dict(),
            }
        return {
            "success": True,
            "message": "Trunk open degree set successfully",
            "current_state": self.to_dict(),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert the rear trunk to a dictionary representation."""
        return {
            "is_on": {
                "value": self.is_on,
                "description": "Whether the rear trunk system is on",
                "type": type(self.is_on).__name__,
            },
            "open_degree": {
                "value": self.open_degree,
                "description": "Trunk open degree (0-100)",
                "type": type(self.open_degree).__name__,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Trunk":
        """Create an instance from a dictionary representation."""
        instance = cls()
        instance.is_on = data["is_on"]["value"]
        instance.open_degree = data["open_degree"]["value"]
        return instance
