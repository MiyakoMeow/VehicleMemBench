from ..utils import api
from typing import Dict, Any


class FrontTrunk:
    """
    Entity class representing the front trunk.
    """

    PARAMS_DESCRIPTION = {
        "switch": {
            "description": "Power state of the front trunk (frunk) system that controls whether the trunk can be operated.",
            "type": "bool",
            "required": True,
            "valid_values": [True, False],
            "value_meanings": {
                True: "Activate the front trunk system - enables trunk operation and allows setting open degree",
                False: "Deactivate the front trunk system - disables trunk operation",
            },
            "default": False,
            "notes": "The front trunk (frunk) is a storage compartment located at the front of electric vehicles where a traditional engine would be. When open_degree is set to a value > 0, the system automatically activates. When open_degree is set to 0, the system automatically deactivates.",
        },
        "degree": {
            "description": "Opening angle/degree of the front trunk lid, controlling how far the trunk is opened.",
            "type": "int",
            "required": True,
            "valid_range": {"min": 0, "max": 100},
            "unit": "percentage (%)",
            "value_meanings": {
                0: "Fully closed - trunk lid is completely shut (also automatically turns off the system)",
                50: "Half open - trunk lid is partially raised",
                100: "Fully open - trunk lid is completely raised for maximum access",
            },
            "default": 0,
            "notes": "Setting a degree > 0 will automatically activate the front trunk system (is_on = True). Setting degree to 0 will automatically deactivate the system (is_on = False). This allows single-command operation without needing to explicitly turn on/off the system.",
        },
    }

    def __init__(self):
        """Initialize with default values."""
        self._is_on = False
        self._open_degree = 0

    # === POWER STATE ===
    @property
    def is_on(self) -> bool:
        """Get front trunk state."""
        return self._is_on

    @is_on.setter
    def is_on(self, value: bool):
        """Set front trunk state."""
        self._is_on = bool(value)

    # === OPEN DEGREE ===
    @property
    def open_degree(self) -> int:
        """Get open degree (0-100)."""
        return self._open_degree

    @open_degree.setter
    def open_degree(self, value: int):
        """Set open degree (0-100). Auto turns on if off, auto turns off if degree is 0."""
        value = self._validate_open_degree(value)
        if value > 0 and not self._is_on:
            self._is_on = True
        self._open_degree = value
        if value == 0:
            self._is_on = False

    def _validate_open_degree(self, value: int) -> int:
        if not isinstance(value, int):
            raise ValueError("open_degree must be an integer")
        if value < 0 or value > 100:
            raise ValueError("open_degree must be between 0 and 100")
        return value

    # === API IMPLEMENTATION METHODS ===
    @api("frontTrunk")
    def carcontrol_frontTrunk_switch(self, switch: bool) -> Dict[str, Any]:
        """Turn the front trunk system on or off."""
        self.is_on = switch
        return {
            "success": True,
            "message": f"Front trunk {'activated' if switch else 'deactivated'} successfully",
            "current_state": self.to_dict(),
        }

    @api("frontTrunk")
    def carcontrol_frontTrunk_set_open_degree(self, degree: int) -> Dict[str, Any]:
        """Set front trunk open degree (0-100)."""
        try:
            self.open_degree = degree
        except ValueError as exc:
            return {
                "success": False,
                "message": str(exc),
                "current_state": self.to_dict(),
            }
        return {
            "success": True,
            "message": "Front trunk open degree set successfully",
            "current_state": self.to_dict(),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert the front trunk to a dictionary representation."""
        return {
            "is_on": {
                "value": self.is_on,
                "description": "Whether the front trunk system is on",
                "type": type(self.is_on).__name__,
            },
            "open_degree": {
                "value": self.open_degree,
                "description": "Front trunk open degree (0-100)",
                "type": type(self.open_degree).__name__,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FrontTrunk":
        """Create an instance from a dictionary representation."""
        instance = cls()
        instance.is_on = data["is_on"]["value"]
        instance.open_degree = data["open_degree"]["value"]
        return instance
