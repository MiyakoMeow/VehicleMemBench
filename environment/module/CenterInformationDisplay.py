from ..utils import api
from enum import Enum
from typing import Dict, Any


class CenterInformationDisplay:
    """
    Entity class representing the center information display of a vehicle,
    providing APIs to control display settings like brightness, language, and time format.
    """

    class TimeFormat(Enum):
        FORMAT_12H = "12h"
        FORMAT_24H = "24h"

    class Language(Enum):
        CHINESE = "Chinese"
        ENGLISH = "English"

    PARAMS_DESCRIPTION = {
        "is_on": {
            "description": "Power state of the center information display screen.",
            "type": "bool",
            "required": True,
            "valid_values": [True, False],
            "value_meanings": {
                True: "Turn on the display - activates the screen and enables all display functions",
                False: "Turn off the display - deactivates the screen to save power",
            },
            "default": False,
            "notes": "When other settings (brightness, language, etc.) are modified, the display will be automatically turned on if it was off.",
        },
        "brightness": {
            "description": "Brightness level of the center information display screen.",
            "type": "int",
            "required": True,
            "valid_range": {"min": 0, "max": 100},
            "unit": "percentage (%)",
            "value_meanings": {
                0: "Minimum brightness (screen nearly dark)",
                50: "Medium brightness (balanced visibility)",
                100: "Maximum brightness (full illumination)",
            },
            "default": 30,
            "notes": "Higher brightness improves visibility in bright conditions but consumes more power. The display will be automatically turned on if it was off.",
        },
        "auto_brightness_enabled": {
            "description": "Automatic brightness adjustment mode that adapts screen brightness based on ambient light conditions.",
            "type": "bool",
            "required": True,
            "valid_values": [True, False],
            "value_meanings": {
                True: "Enable auto brightness - screen brightness adjusts automatically based on ambient light sensor",
                False: "Disable auto brightness - use manual brightness setting only",
            },
            "default": False,
            "notes": "When enabled, the system uses ambient light sensors to optimize screen visibility and reduce eye strain. The display will be automatically turned on if it was off.",
        },
        "time_format": {
            "description": "Time display format shown on the center information display.",
            "type": "str",
            "required": True,
            "valid_values": ["12h", "24h"],
            "value_meanings": {
                "12h": "12-hour format with AM/PM indicator (e.g., 2:30 PM)",
                "24h": "24-hour format without AM/PM (e.g., 14:30)",
            },
            "default": "24h",
            "notes": "This setting affects how time is displayed throughout the infotainment system. The display will be automatically turned on if it was off.",
        },
        "language": {
            "description": "Display language for the center information display user interface.",
            "type": "str",
            "required": True,
            "valid_values": ["Chinese", "English"],
            "value_meanings": {
                "Chinese": "Display all text and menus in Simplified Chinese (简体中文)",
                "English": "Display all text and menus in English",
            },
            "default": "English",
            "notes": "Changing the language will update all system menus, notifications, and text displays. The display will be automatically turned on if it was off.",
        },
    }

    def __init__(self):
        """Initialize with default values."""
        self._is_on = False
        self._brightness = 30
        self._auto_brightness_enabled = False
        self._time_format = self.TimeFormat.FORMAT_24H
        self._language = self.Language.ENGLISH

    # === POWER STATE ===
    @property
    def is_on(self) -> bool:
        """Get display power state."""
        return self._is_on

    @is_on.setter
    def is_on(self, value: bool):
        """Set display power state."""
        self._is_on = bool(value)

    def _ensure_on(self) -> None:
        """Auto turn on display if not already on."""
        if not self._is_on:
            self._is_on = True

    # === BRIGHTNESS ===
    @property
    def brightness(self) -> int:
        """Get current brightness (0-100)."""
        return self._brightness

    @brightness.setter
    def brightness(self, value: int):
        """Set brightness (0-100)."""
        self._brightness = self._validate_brightness(value)

    # === AUTO BRIGHTNESS ===
    @property
    def auto_brightness_enabled(self) -> bool:
        """Get auto brightness mode state."""
        return self._auto_brightness_enabled

    @auto_brightness_enabled.setter
    def auto_brightness_enabled(self, value: bool):
        """Set auto brightness mode state."""
        self._auto_brightness_enabled = bool(value)

    # === TIME FORMAT ===
    @property
    def time_format(self) -> "CenterInformationDisplay.TimeFormat":
        """Get time format (12h/24h)."""
        return self._time_format

    @time_format.setter
    def time_format(self, value: "CenterInformationDisplay.TimeFormat"):
        """Set time format (12h/24h)."""
        self._time_format = self._validate_time_format(value)

    # === LANGUAGE ===
    @property
    def language(self) -> "CenterInformationDisplay.Language":
        """Get display language."""
        return self._language

    @language.setter
    def language(self, value: "CenterInformationDisplay.Language"):
        """Set display language."""
        self._language = self._validate_language(value)

    def _validate_brightness(self, value: int) -> int:
        if not isinstance(value, int):
            raise ValueError("brightness must be an integer")
        if value < 0 or value > 100:
            raise ValueError("brightness must be between 0 and 100")
        return value

    def _validate_time_format(self, value: "CenterInformationDisplay.TimeFormat") -> "CenterInformationDisplay.TimeFormat":
        if isinstance(value, self.TimeFormat):
            return value
        if isinstance(value, str):
            for item in self.TimeFormat:
                if item.value == value:
                    return item
        raise ValueError("time_format must be '12h' or '24h'")

    def _validate_language(self, value: "CenterInformationDisplay.Language") -> "CenterInformationDisplay.Language":
        if isinstance(value, self.Language):
            return value
        if isinstance(value, str):
            for item in self.Language:
                if item.value == value:
                    return item
        raise ValueError("language must be 'Chinese' or 'English'")

    # === API IMPLEMENTATION METHODS ===
    @api("centerInformationDisplay")
    def carcontrol_centerInformationDisplay_set_power(self, is_on: bool) -> Dict[str, Any]:
        """Set center display power state."""
        self.is_on = is_on
        return {
            "success": True,
            "message": f"Center display {'turned on' if is_on else 'turned off'} successfully",
            "current_state": self.to_dict(),
        }

    @api("centerInformationDisplay")
    def carcontrol_centerInformationDisplay_set_brightness(self, brightness: int) -> Dict[str, Any]:
        """Set center display brightness (0-100)."""
        try:
            self._ensure_on()
            self.brightness = brightness
        except ValueError as exc:
            return {
                "success": False,
                "message": str(exc),
                "current_state": self.to_dict(),
            }
        return {
            "success": True,
            "message": "Center display brightness set successfully",
            "current_state": self.to_dict(),
        }

    @api("centerInformationDisplay")
    def carcontrol_centerInformationDisplay_set_auto_brightness(self, enabled: bool) -> Dict[str, Any]:
        """Enable or disable auto brightness mode."""
        self._ensure_on()
        self.auto_brightness_enabled = enabled
        return {
            "success": True,
            "message": f"Auto brightness {'enabled' if self.auto_brightness_enabled else 'disabled'} successfully",
            "current_state": self.to_dict(),
        }

    @api("centerInformationDisplay")
    def carcontrol_centerInformationDisplay_set_time_format(self, time_format: str) -> Dict[str, Any]:
        """Set time format ('12h' or '24h')."""
        try:
            self._ensure_on()
            self.time_format = time_format
        except ValueError as exc:
            return {
                "success": False,
                "message": str(exc),
                "current_state": self.to_dict(),
            }
        return {
            "success": True,
            "message": "Time format set successfully",
            "current_state": self.to_dict(),
        }

    @api("centerInformationDisplay")
    def carcontrol_centerInformationDisplay_set_language(self, language: str) -> Dict[str, Any]:
        """Set display language ('Chinese' or 'English')."""
        try:
            self._ensure_on()
            self.language = language
        except ValueError as exc:
            return {
                "success": False,
                "message": str(exc),
                "current_state": self.to_dict(),
            }
        return {
            "success": True,
            "message": "Display language set successfully",
            "current_state": self.to_dict(),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert the display settings to a dictionary representation."""
        return {
            "is_on": {
                "value": self.is_on,
                "description": "Whether the center display is turned on",
                "type": type(self.is_on).__name__,
            },
            "brightness": {
                "value": self.brightness,
                "description": "Center display brightness (0-100)",
                "type": type(self.brightness).__name__,
            },
            "auto_brightness_enabled": {
                "value": self.auto_brightness_enabled,
                "description": "Whether auto brightness mode is enabled",
                "type": type(self.auto_brightness_enabled).__name__,
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
    def from_dict(cls, data: Dict[str, Any]) -> "CenterInformationDisplay":
        """Create an instance from a dictionary representation."""
        instance = cls()
        instance.is_on = data["is_on"]["value"]
        instance.brightness = data["brightness"]["value"]
        instance.auto_brightness_enabled = data["auto_brightness_enabled"]["value"]
        instance.time_format = data["time_format"]["value"]
        instance.language = data["language"]["value"]
        return instance
