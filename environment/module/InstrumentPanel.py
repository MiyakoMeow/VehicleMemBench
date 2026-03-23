from ..utils import api
from enum import Enum
from typing import Dict, Any


class InstrumentPanel:
    """
    Entity class representing a vehicle instrument panel.
    """

    class Theme(Enum):
        SCENE = "scene"
        MAP = "map"

    class Color(Enum):
        YELLOW = "yellow"
        GREEN = "green"
        RED = "red"
        ORANGE = "orange"
        WHITE = "white"
        BLACK = "black"

    class BehaviorMode(Enum):
        CONSTANT = "constant"
        BLINK = "blink"
        OFF = "off"

    class TimeFormat(Enum):
        FORMAT_12H = "12h"
        FORMAT_24H = "24h"

    class Language(Enum):
        CHINESE = "Chinese"
        ENGLISH = "English"

    PARAMS_DESCRIPTION = {
        "mileage": {
            "description": "Total accumulated mileage/odometer reading of the vehicle displayed on the instrument panel.",
            "type": "int",
            "required": True,
            "valid_range": {"min": 1, "max": None},
            "unit": "kilometers (km)",
            "default": 1,
            "notes": "Must be greater than 0. This is typically a read-only value that accumulates over time, but can be set for simulation purposes.",
        },
        "theme": {
            "description": "Visual theme/layout of the instrument panel display.",
            "type": "str",
            "required": True,
            "valid_values": ["scene", "map"],
            "value_meanings": {
                "scene": "Scene mode - traditional instrument layout showing speedometer, tachometer, and gauges",
                "map": "Map mode - displays navigation map as the primary view with driving info overlaid",
            },
            "default": "scene",
            "notes": "Map mode is useful during navigation for a larger map view. Scene mode provides classic instrument aesthetics.",
        },
        "brightness": {
            "description": "Brightness level of the instrument panel display.",
            "type": "int",
            "required": True,
            "valid_range": {"min": 1, "max": 5},
            "value_meanings": {
                1: "Dimmest - minimal brightness for night driving",
                3: "Medium - balanced brightness (default)",
                5: "Brightest - maximum brightness for daylight",
            },
            "default": 3,
            "notes": "Lower values recommended at night to reduce eye strain and glare. Can be overridden by auto_brightness if enabled.",
        },
        "color": {
            "description": "Primary display color theme of the instrument panel.",
            "type": "str",
            "required": True,
            "valid_values": ["yellow", "green", "red", "orange", "white", "black"],
            "value_meanings": {
                "yellow": "Yellow color theme - warm, high visibility",
                "green": "Green color theme - eco/nature style",
                "red": "Red color theme - sporty/aggressive style",
                "orange": "Orange color theme - warm accent",
                "white": "White color theme - clean, modern look (default)",
                "black": "Black color theme - dark mode style",
            },
            "default": "white",
            "notes": "Color affects the overall visual appearance of gauges, text, and UI elements on the instrument panel.",
        },
        "mode": {
            "description": "Display behavior mode that controls how the instrument panel content is shown.",
            "type": "str",
            "required": True,
            "valid_values": ["constant", "blink", "off"],
            "value_meanings": {
                "constant": "Constant mode - display stays on continuously with stable content",
                "blink": "Blink mode - display flashes/blinks to draw attention (used for warnings)",
                "off": "Off mode - display is turned off",
            },
            "default": "constant",
            "notes": "Blink mode is typically triggered automatically for critical warnings. Off mode can be used to disable the display entirely.",
        },
        "enabled": {
            "description": "Automatic brightness adjustment that adapts display brightness based on ambient light.",
            "type": "bool",
            "required": True,
            "valid_values": [True, False],
            "value_meanings": {
                True: "Enable auto brightness - system adjusts brightness automatically based on light sensor",
                False: "Disable auto brightness - use manual brightness setting only",
            },
            "default": True,
            "notes": "When enabled, the system uses ambient light sensors to optimize display visibility. Manual brightness setting is ignored when auto brightness is active.",
        },
        "time_format": {
            "description": "Time display format shown on the instrument panel clock.",
            "type": "str",
            "required": True,
            "valid_values": ["12h", "24h"],
            "value_meanings": {
                "12h": "12-hour format with AM/PM indicator (e.g., 2:30 PM)",
                "24h": "24-hour format without AM/PM (e.g., 14:30)",
            },
            "default": "24h",
            "notes": "Affects how time is displayed on the instrument panel clock and any time-related information.",
        },
        "language": {
            "description": "Display language for text and labels on the instrument panel.",
            "type": "str",
            "required": True,
            "valid_values": ["Chinese", "English"],
            "value_meanings": {
                "Chinese": "Display all text in Simplified Chinese (简体中文)",
                "English": "Display all text in English",
            },
            "default": "English",
            "notes": "Affects warning messages, labels, menu text, and all other textual content on the instrument panel.",
        },
    }

    def __init__(self):
        """Initialize with default values."""
        self._total_mileage = 1
        self._theme = self.Theme.SCENE
        self._brightness = 3
        self._color = self.Color.WHITE
        self._behavior_mode = self.BehaviorMode.CONSTANT
        self._auto_brightness_enabled = False
        self._time_format = self.TimeFormat.FORMAT_24H
        self._language = self.Language.ENGLISH

    # === TOTAL MILEAGE ===
    @property
    def total_mileage(self) -> int:
        """Get total mileage (>0)."""
        return self._total_mileage

    @total_mileage.setter
    def total_mileage(self, value: int):
        """Set total mileage (>0)."""
        self._total_mileage = self._validate_total_mileage(value)

    # === THEME ===
    @property
    def theme(self) -> "InstrumentPanel.Theme":
        """Get current theme."""
        return self._theme

    @theme.setter
    def theme(self, value: "InstrumentPanel.Theme"):
        """Set current theme."""
        self._theme = self._validate_theme(value)

    # === BRIGHTNESS ===
    @property
    def brightness(self) -> int:
        """Get brightness (1-5)."""
        return self._brightness

    @brightness.setter
    def brightness(self, value: int):
        """Set brightness (1-5)."""
        self._brightness = self._validate_brightness(value)

    # === COLOR ===
    @property
    def color(self) -> "InstrumentPanel.Color":
        """Get display color."""
        return self._color

    @color.setter
    def color(self, value: "InstrumentPanel.Color"):
        """Set display color."""
        self._color = self._validate_color(value)

    # === BEHAVIOR MODE ===
    @property
    def behavior_mode(self) -> "InstrumentPanel.BehaviorMode":
        """Get behavior mode."""
        return self._behavior_mode

    @behavior_mode.setter
    def behavior_mode(self, value: "InstrumentPanel.BehaviorMode"):
        """Set behavior mode."""
        self._behavior_mode = self._validate_behavior_mode(value)

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
    def time_format(self) -> "InstrumentPanel.TimeFormat":
        """Get time format (12h/24h)."""
        return self._time_format

    @time_format.setter
    def time_format(self, value: "InstrumentPanel.TimeFormat"):
        """Set time format (12h/24h)."""
        self._time_format = self._validate_time_format(value)

    # === LANGUAGE ===
    @property
    def language(self) -> "InstrumentPanel.Language":
        """Get display language."""
        return self._language

    @language.setter
    def language(self, value: "InstrumentPanel.Language"):
        """Set display language."""
        self._language = self._validate_language(value)

    def _validate_total_mileage(self, value: int) -> int:
        if not isinstance(value, int):
            raise ValueError("total_mileage must be an integer")
        if value <= 0:
            raise ValueError("total_mileage must be greater than 0")
        return value

    def _validate_brightness(self, value: int) -> int:
        if not isinstance(value, int):
            raise ValueError("brightness must be an integer")
        if value < 1 or value > 5:
            raise ValueError("brightness must be between 1 and 5")
        return value

    def _validate_theme(self, value: "InstrumentPanel.Theme") -> "InstrumentPanel.Theme":
        if isinstance(value, self.Theme):
            return value
        if isinstance(value, str):
            for item in self.Theme:
                if item.value == value:
                    return item
        raise ValueError("theme must be 'scene' or 'map'")

    def _validate_color(self, value: "InstrumentPanel.Color") -> "InstrumentPanel.Color":
        if isinstance(value, self.Color):
            return value
        if isinstance(value, str):
            for item in self.Color:
                if item.value == value:
                    return item
        raise ValueError("color is invalid")

    def _validate_behavior_mode(self, value: "InstrumentPanel.BehaviorMode") -> "InstrumentPanel.BehaviorMode":
        if isinstance(value, self.BehaviorMode):
            return value
        if isinstance(value, str):
            for item in self.BehaviorMode:
                if item.value == value:
                    return item
        raise ValueError("behavior_mode must be 'constant', 'blink', or 'off'")

    def _validate_time_format(self, value: "InstrumentPanel.TimeFormat") -> "InstrumentPanel.TimeFormat":
        if isinstance(value, self.TimeFormat):
            return value
        if isinstance(value, str):
            for item in self.TimeFormat:
                if item.value == value:
                    return item
        raise ValueError("time_format must be '12h' or '24h'")

    def _validate_language(self, value: "InstrumentPanel.Language") -> "InstrumentPanel.Language":
        if isinstance(value, self.Language):
            return value
        if isinstance(value, str):
            for item in self.Language:
                if item.value == value:
                    return item
        raise ValueError("language must be 'Chinese' or 'English'")

    # === API IMPLEMENTATION METHODS ===
    @api("instrumentPanel")
    def carcontrol_instrumentPanel_set_total_mileage(self, mileage: int) -> Dict[str, Any]:
        """Set total mileage (>0)."""
        try:
            self.total_mileage = mileage
        except ValueError as exc:
            return {
                "success": False,
                "message": str(exc),
                "current_state": self.to_dict(),
            }
        return {
            "success": True,
            "message": "Total mileage set successfully",
            "current_state": self.to_dict(),
        }

    @api("instrumentPanel")
    def carcontrol_instrumentPanel_set_theme(self, theme: str) -> Dict[str, Any]:
        """Set instrument panel theme ('scene' or 'map')."""
        try:
            self.theme = theme
        except ValueError as exc:
            return {
                "success": False,
                "message": str(exc),
                "current_state": self.to_dict(),
            }
        return {
            "success": True,
            "message": "Instrument panel theme set successfully",
            "current_state": self.to_dict(),
        }

    @api("instrumentPanel")
    def carcontrol_instrumentPanel_set_brightness(self, brightness: int) -> Dict[str, Any]:
        """Set brightness (1-5)."""
        try:
            self.brightness = brightness
        except ValueError as exc:
            return {
                "success": False,
                "message": str(exc),
                "current_state": self.to_dict(),
            }
        return {
            "success": True,
            "message": "Brightness set successfully",
            "current_state": self.to_dict(),
        }

    @api("instrumentPanel")
    def carcontrol_instrumentPanel_set_color(self, color: str) -> Dict[str, Any]:
        """Set display color."""
        try:
            self.color = color
        except ValueError as exc:
            return {
                "success": False,
                "message": str(exc),
                "current_state": self.to_dict(),
            }
        return {
            "success": True,
            "message": "Display color set successfully",
            "current_state": self.to_dict(),
        }

    @api("instrumentPanel")
    def carcontrol_instrumentPanel_set_behavior_mode(self, mode: str) -> Dict[str, Any]:
        """Set behavior mode ('constant', 'blink', or 'off')."""
        try:
            self.behavior_mode = mode
        except ValueError as exc:
            return {
                "success": False,
                "message": str(exc),
                "current_state": self.to_dict(),
            }
        return {
            "success": True,
            "message": "Behavior mode set successfully",
            "current_state": self.to_dict(),
        }

    @api("instrumentPanel")
    def carcontrol_instrumentPanel_set_auto_brightness(self, enabled: bool) -> Dict[str, Any]:
        """Enable or disable auto brightness mode."""
        self.auto_brightness_enabled = enabled
        return {
            "success": True,
            "message": f"Auto brightness {'enabled' if self.auto_brightness_enabled else 'disabled'} successfully",
            "current_state": self.to_dict(),
        }

    @api("instrumentPanel")
    def carcontrol_instrumentPanel_set_time_format(self, time_format: str) -> Dict[str, Any]:
        """Set time format ('12h' or '24h')."""
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
            "message": "Time format set successfully",
            "current_state": self.to_dict(),
        }

    @api("instrumentPanel")
    def carcontrol_instrumentPanel_set_language(self, language: str) -> Dict[str, Any]:
        """Set display language ('Chinese' or 'English')."""
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
            "message": "Display language set successfully",
            "current_state": self.to_dict(),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert the instrument panel to a dictionary representation."""
        return {
            "total_mileage": {
                "value": self.total_mileage,
                "description": "Total vehicle mileage (>0)",
                "type": type(self.total_mileage).__name__,
            },
            "theme": {
                "value": self.theme.value,
                "description": "Instrument panel theme",
                "type": type(self.theme.value).__name__,
            },
            "brightness": {
                "value": self.brightness,
                "description": "Instrument panel brightness (1-5)",
                "type": type(self.brightness).__name__,
            },
            "color": {
                "value": self.color.value,
                "description": "Instrument panel color",
                "type": type(self.color.value).__name__,
            },
            "behavior_mode": {
                "value": self.behavior_mode.value,
                "description": "Instrument panel behavior mode",
                "type": type(self.behavior_mode.value).__name__,
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
    def from_dict(cls, data: Dict[str, Any]) -> "InstrumentPanel":
        """Create an instance from a dictionary representation."""
        instance = cls()
        instance.total_mileage = data["total_mileage"]["value"]
        instance.theme = data["theme"]["value"]
        instance.brightness = data["brightness"]["value"]
        instance.color = data["color"]["value"]
        instance.behavior_mode = data["behavior_mode"]["value"]
        instance.auto_brightness_enabled = data["auto_brightness_enabled"]["value"]
        instance.time_format = data["time_format"]["value"]
        instance.language = data["language"]["value"]
        return instance
