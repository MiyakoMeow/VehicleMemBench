from ..utils import api
from typing import Dict, Any


class HUD:
    """
    Head-Up Display (HUD) entity class that manages the state and operations
    of a vehicle's windshield projection system displaying driving information.
    """

    PARAMS_DESCRIPTION = {
        "switch": {
            "description": "Power state of the Head-Up Display (HUD) system that projects driving information onto the windshield.",
            "type": "bool",
            "required": True,
            "valid_values": [True, False],
            "value_meanings": {
                True: "Turn on the HUD - activates the windshield projection to display speed, navigation, and other driving info",
                False: "Turn off the HUD - deactivates the projection display",
            },
            "default": False,
            "notes": "When height_level or brightness_level is adjusted, the HUD will automatically turn on if it was off.",
        },
        "height_level": {
            "description": "Vertical position of the HUD projection on the windshield, adjustable to match the driver's eye level.",
            "type": "int",
            "required": True,
            "valid_range": {"min": 1, "max": 10},
            "value_meanings": {
                1: "Lowest position - projection appears near the dashboard",
                5: "Middle position - default balanced height",
                10: "Highest position - projection appears higher in the driver's line of sight",
            },
            "default": 5,
            "notes": "Adjust based on driver's seated height and eye position for optimal visibility. Setting this value will automatically turn on the HUD if it was off.",
        },
        "brightness_level": {
            "description": "Brightness intensity of the HUD projection, adjustable based on ambient lighting conditions.",
            "type": "int",
            "required": True,
            "valid_range": {"min": 1, "max": 10},
            "value_meanings": {
                1: "Dimmest setting - suitable for night driving to avoid glare",
                5: "Medium brightness - default balanced setting",
                10: "Brightest setting - suitable for bright sunlight conditions",
            },
            "default": 5,
            "notes": "Lower brightness recommended at night to reduce eye strain. Higher brightness needed in direct sunlight for visibility. Setting this value will automatically turn on the HUD if it was off.",
        },
    }

    def __init__(self):
        """Initialize with default values."""
        self._is_on = False
        self._height_level = 5
        self._brightness_level = 5

    # === POWER STATE ===
    @property
    def is_on(self) -> bool:
        """Get current power state."""
        return self._is_on

    @is_on.setter
    def is_on(self, value: bool):
        """Set current power state."""
        self._is_on = value

    # === HEIGHT LEVEL ===
    @property
    def height_level(self) -> int:
        """Get current height level (1-10)."""
        return self._height_level

    @height_level.setter
    def height_level(self, value: int):
        """Set height level (1-10)."""
        self._height_level = self._validate_level(value, "height_level")

    # === BRIGHTNESS LEVEL ===
    @property
    def brightness_level(self) -> int:
        """Get current brightness level (1-10)."""
        return self._brightness_level

    @brightness_level.setter
    def brightness_level(self, value: int):
        """Set brightness level (1-10)."""
        self._brightness_level = self._validate_level(value, "brightness_level")

    def _validate_level(self, value: int, name: str) -> int:
        if not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
        if value < 1 or value > 10:
            raise ValueError(f"{name} must be between 1 and 10")
        return value

    def _ensure_on(self) -> None:
        """Auto turn on HUD if not already on."""
        if not self._is_on:
            self._is_on = True

    # === API IMPLEMENTATION METHODS ===
    @api("HUD")
    def carcontrol_HUD_switch(self, switch: bool) -> Dict[str, Any]:
        """Turn the HUD on or off."""
        self.is_on = switch
        return {
            "success": True,
            "message": f"HUD {'activated' if switch else 'deactivated'} successfully",
            "current_state": self.to_dict(),
        }

    @api("HUD")
    def carcontrol_HUD_set_height_level(self, level: int) -> Dict[str, Any]:
        """Set HUD height level (1-10)."""
        try:
            self._ensure_on()
            self.height_level = level
        except ValueError as exc:
            return {
                "success": False,
                "message": str(exc),
                "current_state": self.to_dict(),
            }
        return {
            "success": True,
            "message": "HUD height level set successfully",
            "current_state": self.to_dict(),
        }

    @api("HUD")
    def carcontrol_HUD_set_brightness_level(self, level: int) -> Dict[str, Any]:
        """Set HUD brightness level (1-10)."""
        try:
            self._ensure_on()
            self.brightness_level = level
        except ValueError as exc:
            return {
                "success": False,
                "message": str(exc),
                "current_state": self.to_dict(),
            }
        return {
            "success": True,
            "message": "HUD brightness level set successfully",
            "current_state": self.to_dict(),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert the HUD to a dictionary representation."""
        return {
            "is_on": {
                "value": self.is_on,
                "description": "Whether the HUD is turned on or off",
                "type": type(self.is_on).__name__,
            },
            "height_level": {
                "value": self.height_level,
                "description": "HUD height level (1-10)",
                "type": type(self.height_level).__name__,
            },
            "brightness_level": {
                "value": self.brightness_level,
                "description": "HUD brightness level (1-10)",
                "type": type(self.brightness_level).__name__,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HUD":
        """Create a HUD instance from a dictionary representation."""
        instance = cls()
        instance.is_on = data["is_on"]["value"]
        instance.height_level = data["height_level"]["value"]
        instance.brightness_level = data["brightness_level"]["value"]
        return instance
