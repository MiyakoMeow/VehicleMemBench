from ..utils import api
from enum import Enum
from typing import Dict, Any


class OverheadScreen:
    """
    Entity class representing the overhead screen.
    """

    class TimeFormat(Enum):
        FORMAT_12H = "12h"
        FORMAT_24H = "24h"

    class Language(Enum):
        CHINESE = "Chinese"
        ENGLISH = "English"

    PARAMS_DESCRIPTION = {
        "switch": {
            "description": "Power state of the overhead screen (ceiling-mounted display for rear passengers).",
            "type": "bool",
            "required": True,
            "valid_values": [True, False],
            "value_meanings": {
                True: "Turn on the overhead screen",
                False: "Turn off the overhead screen",
            },
            "default": False,
            "notes": "All other overhead screen operations will automatically turn on the screen if it was off.",
        },
        "brightness_level": {
            "description": "Brightness level of the overhead screen display.",
            "type": "int",
            "required": True,
            "valid_range": {"min": 0, "max": 5},
            "value_meanings": {
                0: "Screen off / minimum brightness",
                1: "Very low brightness",
                2: "Low brightness",
                3: "Medium brightness",
                4: "High brightness",
                5: "Maximum brightness",
            },
            "default": 0,
            "notes": "System will auto turn on if needed. Lower values recommended for night driving to avoid distracting the driver.",
        },
        "time_format": {
            "description": "Time display format shown on the overhead screen.",
            "type": "str",
            "required": True,
            "valid_values": ["12h", "24h"],
            "value_meanings": {
                "12h": "12-hour format with AM/PM indicator (e.g., 2:30 PM)",
                "24h": "24-hour format without AM/PM (e.g., 14:30)",
            },
            "default": "24h",
            "notes": "System will auto turn on if needed.",
        },
        "language": {
            "description": "Display language for text and menus on the overhead screen.",
            "type": "str",
            "required": True,
            "valid_values": ["Chinese", "English"],
            "value_meanings": {
                "Chinese": "Display all text in Simplified Chinese (简体中文)",
                "English": "Display all text in English",
            },
            "aliases": {
                "中文": "Chinese",
                "英文": "English",
            },
            "default": "Chinese",
            "notes": "System will auto turn on if needed. Chinese aliases are supported.",
        },
    }

    def __init__(self):
        """Initialize with default values."""
        self._is_on = False
        self._brightness_level = 0
        self._time_format = self.TimeFormat.FORMAT_24H
        self._language = self.Language.CHINESE

    # === POWER STATE ===
    @property
    def is_on(self) -> bool:
        """Get current power state."""
        return self._is_on

    @is_on.setter
    def is_on(self, value: bool):
        """Set current power state."""
        self._is_on = bool(value)

    # === BRIGHTNESS LEVEL ===
    @property
    def brightness_level(self) -> int:
        """Get brightness level (0-5)."""
        return self._brightness_level

    @brightness_level.setter
    def brightness_level(self, value: int):
        """Set brightness level (0-5)."""
        self._brightness_level = self._validate_range(value, 0, 5, "brightness_level")

    # === TIME FORMAT ===
    @property
    def time_format(self) -> "OverheadScreen.TimeFormat":
        """Get time format (12h/24h)."""
        return self._time_format

    @time_format.setter
    def time_format(self, value: "OverheadScreen.TimeFormat"):
        """Set time format (12h/24h)."""
        self._time_format = self._validate_time_format(value)

    # === LANGUAGE ===
    @property
    def language(self) -> "OverheadScreen.Language":
        """Get display language."""
        return self._language

    @language.setter
    def language(self, value: "OverheadScreen.Language"):
        """Set display language."""
        self._language = self._validate_language(value)

    def _ensure_on(self) -> None:
        """Auto turn on overhead screen if not already on."""
        if not self._is_on:
            self._is_on = True

    def _validate_range(self, value: int, min_value: int, max_value: int, name: str) -> int:
        if not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
        if value < min_value or value > max_value:
            raise ValueError(f"{name} must be between {min_value} and {max_value}")
        return value

    def _validate_time_format(self, value: "OverheadScreen.TimeFormat") -> "OverheadScreen.TimeFormat":
        if isinstance(value, self.TimeFormat):
            return value
        if isinstance(value, str):
            normalized = value.strip()
            if normalized == "12h":
                return self.TimeFormat.FORMAT_12H
            if normalized == "24h":
                return self.TimeFormat.FORMAT_24H
        raise ValueError("time_format must be '12h' or '24h'")

    def _validate_language(self, value: "OverheadScreen.Language") -> "OverheadScreen.Language":
        if isinstance(value, self.Language):
            return value
        if isinstance(value, str):
            normalized = value.strip()
            if normalized in ("Chinese", "中文"):
                return self.Language.CHINESE
            if normalized in ("English", "英文"):
                return self.Language.ENGLISH
        raise ValueError("language must be 'Chinese' or 'English'")

    # === API IMPLEMENTATION METHODS ===
    @api("overheadScreen")
    def carcontrol_overheadScreen_switch(self, switch: bool) -> Dict[str, Any]:
        """Turn the overhead screen on or off."""
        self.is_on = switch
        return {
            "success": True,
            "message": f"Overhead screen {'activated' if switch else 'deactivated'} successfully",
            "current_state": self.to_dict(),
        }

    @api("overheadScreen")
    def carcontrol_overheadScreen_set_brightness_level(self, level: int) -> Dict[str, Any]:
        """Set overhead screen brightness level (0-5)."""
        self._ensure_on()
        try:
            self.brightness_level = level
        except ValueError as exc:
            return {
                "success": False,
                "message": str(exc),
                "current_state": self.to_dict(),
            }
        return {
            "success": True,
            "message": "Overhead screen brightness level set successfully",
            "current_state": self.to_dict(),
        }

    @api("overheadScreen")
    def carcontrol_overheadScreen_set_time_format(self, time_format: str) -> Dict[str, Any]:
        """Set time format ('12h' or '24h')."""
        self._ensure_on()
        try:
            self.time_format = time_format
        except ValueError as exc:
            return {
                "success": False,
                "message": str(exc),
                "current_state": self.to_dict(),
            }
        return {
            "success": True,
            "message": "Overhead screen time format set successfully",
            "current_state": self.to_dict(),
        }

    @api("overheadScreen")
    def carcontrol_overheadScreen_set_language(self, language: str) -> Dict[str, Any]:
        """Set display language ('Chinese' or 'English')."""
        self._ensure_on()
        try:
            self.language = language
        except ValueError as exc:
            return {
                "success": False,
                "message": str(exc),
                "current_state": self.to_dict(),
            }
        return {
            "success": True,
            "message": "Overhead screen language set successfully",
            "current_state": self.to_dict(),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert overhead screen settings to a dictionary representation."""
        return {
            "is_on": {
                "value": self.is_on,
                "description": "Whether the overhead screen is turned on or off",
                "type": type(self.is_on).__name__,
            },
            "brightness_level": {
                "value": self.brightness_level,
                "description": "Overhead screen brightness level (0-5)",
                "type": type(self.brightness_level).__name__,
            },
            "time_format": {
                "value": self.time_format.value,
                "description": "Time format (12h/24h)",
                "type": type(self.time_format.value).__name__,
            },
            "language": {
                "value": self.language.value,
                "description": "Display language",
                "type": type(self.language.value).__name__,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OverheadScreen":
        """Create an instance from a dictionary representation."""
        instance = cls()
        instance.is_on = data["is_on"]["value"]
        instance.brightness_level = data["brightness_level"]["value"]
        instance.time_format = data["time_format"]["value"]
        instance.language = data["language"]["value"]
        return instance
