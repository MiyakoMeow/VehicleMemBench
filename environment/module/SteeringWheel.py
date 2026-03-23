from ..utils import api
from typing import Dict, Any


class SteeringWheel:
    """
    Entity class representing the steering wheel.
    """

    PARAMS_DESCRIPTION = {
        "view_display_enabled": {
            "description": "Steering wheel mounted display screen that shows vehicle information.",
            "type": "bool",
            "required": True,
            "valid_values": [True, False],
            "value_meanings": {
                True: "Enable steering wheel display - shows speed, navigation, media info",
                False: "Disable steering wheel display",
            },
            "default": False,
            "notes": "Some vehicles have a small display on the steering wheel for quick info access.",
        },
        "heating_enabled": {
            "description": "Steering wheel heating function to warm the wheel surface.",
            "type": "bool",
            "required": True,
            "valid_values": [True, False],
            "value_meanings": {
                True: "Enable steering wheel heating",
                False: "Disable steering wheel heating",
            },
            "default": False,
            "notes": "Recommended in cold weather for comfort. Will be auto-enabled when setting heating level.",
        },
        "heating_level": {
            "description": "Intensity level of steering wheel heating.",
            "type": "int",
            "required": True,
            "valid_range": {"min": 0, "max": 10},
            "value_meanings": {
                0: "No heating",
                1: "Minimum heating",
                5: "Medium heating",
                10: "Maximum heating",
            },
            "default": 0,
            "notes": "System will auto enable heating if needed. Higher levels provide more warmth.",
        },
    }

    def __init__(self):
        """Initialize with default values."""
        self._view_display_enabled = False
        self._heating_enabled = False
        self._heating_level = 0

    # === VIEW DISPLAY ===
    @property
    def view_display_enabled(self) -> bool:
        """Get steering wheel view display state."""
        return self._view_display_enabled

    @view_display_enabled.setter
    def view_display_enabled(self, value: bool):
        """Set steering wheel view display state."""
        self._view_display_enabled = bool(value)

    # === HEATING ENABLED ===
    @property
    def heating_enabled(self) -> bool:
        """Get steering wheel heating state."""
        return self._heating_enabled

    @heating_enabled.setter
    def heating_enabled(self, value: bool):
        """Set steering wheel heating state."""
        self._heating_enabled = bool(value)

    # === HEATING LEVEL ===
    @property
    def heating_level(self) -> int:
        """Get steering wheel heating level (0-10)."""
        return self._heating_level

    @heating_level.setter
    def heating_level(self, value: int):
        """Set steering wheel heating level (0-10)."""
        self._heating_level = self._validate_range(value, 0, 10, "heating_level")

    def _ensure_heating_on(self) -> None:
        """Auto enable heating if not already enabled."""
        if not self._heating_enabled:
            self._heating_enabled = True

    def _validate_range(self, value: int, min_value: int, max_value: int, name: str) -> int:
        if not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
        if value < min_value or value > max_value:
            raise ValueError(f"{name} must be between {min_value} and {max_value}")
        return value

    # === API IMPLEMENTATION METHODS ===
    @api("steeringWheel")
    def carcontrol_steeringWheel_set_view_display_enabled(self, enabled: bool) -> Dict[str, Any]:
        """Enable or disable steering wheel view display."""
        self.view_display_enabled = enabled
        return {
            "success": True,
            "message": "Steering wheel view display set successfully",
            "current_state": self.to_dict(),
        }

    @api("steeringWheel")
    def carcontrol_steeringWheel_set_heating_enabled(self, enabled: bool) -> Dict[str, Any]:
        """Enable or disable steering wheel heating."""
        self.heating_enabled = enabled
        return {
            "success": True,
            "message": "Steering wheel heating set successfully",
            "current_state": self.to_dict(),
        }

    @api("steeringWheel")
    def carcontrol_steeringWheel_set_heating_level(self, level: int) -> Dict[str, Any]:
        """Set steering wheel heating level (0-10)."""
        self._ensure_heating_on()
        try:
            self.heating_level = level
        except ValueError as exc:
            return {
                "success": False,
                "message": str(exc),
                "current_state": self.to_dict(),
            }
        return {
            "success": True,
            "message": "Steering wheel heating level set successfully",
            "current_state": self.to_dict(),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert steering wheel settings to a dictionary representation."""
        return {
            "view_display_enabled": {
                "value": self.view_display_enabled,
                "description": "Whether steering wheel view display is enabled",
                "type": type(self.view_display_enabled).__name__,
            },
            "heating_enabled": {
                "value": self.heating_enabled,
                "description": "Whether steering wheel heating is enabled",
                "type": type(self.heating_enabled).__name__,
            },
            "heating_level": {
                "value": self.heating_level,
                "description": "Steering wheel heating level (0-10)",
                "type": type(self.heating_level).__name__,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SteeringWheel":
        """Create an instance from a dictionary representation."""
        instance = cls()
        instance.view_display_enabled = data["view_display_enabled"]["value"]
        instance.heating_enabled = data["heating_enabled"]["value"]
        instance.heating_level = data["heating_level"]["value"]
        return instance
