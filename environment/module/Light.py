from ..utils import api
from typing import Dict, Any, List


class Light:
    """
    Entity class representing vehicle lights.
    """

    PARAMS_DESCRIPTION = {
        "fog_light": {
            "description": "Front fog lights that improve visibility in foggy, rainy, or dusty conditions.",
            "type": "bool",
            "required": True,
            "valid_values": [True, False],
            "value_meanings": {
                True: "Turn on fog lights - activates low-mounted lights for better visibility in poor weather",
                False: "Turn off fog lights",
            },
            "default": False,
            "notes": "Fog lights are positioned lower than headlights to reduce glare from fog/mist reflection.",
        },
        "high_beam": {
            "description": "High beam headlights for maximum forward illumination on dark roads.",
            "type": "bool",
            "required": True,
            "valid_values": [True, False],
            "value_meanings": {
                True: "Turn on high beam - provides maximum long-range illumination",
                False: "Turn off high beam",
            },
            "default": False,
            "notes": "High beam should be turned off when approaching other vehicles to avoid blinding other drivers.",
        },
        "low_beam_enabled": {
            "description": "Low beam headlights for standard nighttime driving illumination.",
            "type": "bool",
            "required": True,
            "valid_values": [True, False],
            "value_meanings": {
                True: "Turn on low beam - standard headlights for night driving",
                False: "Turn off low beam",
            },
            "default": False,
            "notes": "Low beam must be turned on before adjusting the brightness level.",
        },
        "low_beam_level": {
            "description": "Brightness/intensity level of the low beam headlights.",
            "type": "str",
            "required": True,
            "valid_values": ["lowest", "low", "medium", "high", "highest"],
            "value_meanings": {
                "lowest": "Minimum brightness - suitable for well-lit areas",
                "low": "Low brightness",
                "medium": "Medium brightness - balanced setting",
                "high": "High brightness",
                "highest": "Maximum brightness - for very dark conditions",
            },
            "aliases": {
                "最低": "lowest",
                "低": "low",
                "中": "medium",
                "高": "high",
                "最高": "highest",
            },
            "default": "lowest",
            "notes": "Low beam must be on to adjust level. Chinese aliases are supported.",
        },
        "daytime_running": {
            "description": "Daytime running lights (DRL) that increase vehicle visibility during daytime.",
            "type": "bool",
            "required": True,
            "valid_values": [True, False],
            "value_meanings": {
                True: "Turn on DRL - improves vehicle visibility to other drivers during daytime",
                False: "Turn off DRL",
            },
            "default": False,
            "notes": "DRL automatically makes the vehicle more visible during daytime driving for safety.",
        },
        "left_turn": {
            "description": "Left turn signal indicator light.",
            "type": "bool",
            "required": True,
            "valid_values": [True, False],
            "value_meanings": {
                True: "Activate left turn signal - flashing indicator for left turns or lane changes",
                False: "Deactivate left turn signal",
            },
            "default": False,
            "notes": "Turn signals should be activated before making turns or changing lanes to alert other drivers.",
        },
        "right_turn": {
            "description": "Right turn signal indicator light.",
            "type": "bool",
            "required": True,
            "valid_values": [True, False],
            "value_meanings": {
                True: "Activate right turn signal - flashing indicator for right turns or lane changes",
                False: "Deactivate right turn signal",
            },
            "default": False,
            "notes": "Turn signals should be activated before making turns or changing lanes to alert other drivers.",
        },
        "auto_headlight": {
            "description": "Automatic headlight system that turns headlights on/off based on ambient light.",
            "type": "bool",
            "required": True,
            "valid_values": [True, False],
            "value_meanings": {
                True: "Enable auto headlight - system automatically controls headlights based on light sensor",
                False: "Disable auto headlight - manual headlight control only",
            },
            "default": False,
            "notes": "When enabled, headlights automatically turn on in tunnels, at dusk, or in dark conditions.",
        },
        "hazard": {
            "description": "Hazard warning lights (emergency flashers) that flash all turn signals simultaneously.",
            "type": "bool",
            "required": True,
            "valid_values": [True, False],
            "value_meanings": {
                True: "Turn on hazard lights - all turn signals flash to warn other drivers of emergency/hazard",
                False: "Turn off hazard lights",
            },
            "default": False,
            "notes": "Use hazard lights when stopped on roadside, in emergency situations, or to warn of sudden braking.",
        },
        "position": {
            "description": "Position lights (parking lights/sidelights) for low-visibility vehicle presence indication.",
            "type": "bool",
            "required": True,
            "valid_values": [True, False],
            "value_meanings": {
                True: "Turn on position lights - indicates vehicle presence in low light",
                False: "Turn off position lights",
            },
            "default": False,
            "notes": "Position lights are dimmer than headlights, used when parked or in light fog.",
        },
        "tail": {
            "description": "Tail lights at the rear of the vehicle for visibility to following traffic.",
            "type": "bool",
            "required": True,
            "valid_values": [True, False],
            "value_meanings": {
                True: "Turn on tail lights - illuminates rear of vehicle for visibility",
                False: "Turn off tail lights",
            },
            "default": False,
            "notes": "Tail lights typically activate automatically with headlights but can be controlled separately.",
        },
        "ambient_enabled": {
            "description": "Interior ambient/mood lighting for cabin atmosphere enhancement.",
            "type": "bool",
            "required": True,
            "valid_values": [True, False],
            "value_meanings": {
                True: "Turn on ambient light - activates interior decorative lighting",
                False: "Turn off ambient light",
            },
            "default": False,
            "notes": "Ambient light must be on before setting the color. Provides aesthetic interior illumination.",
        },
        "ambient_color": {
            "description": "Color of the interior ambient lighting.",
            "type": "str",
            "required": True,
            "valid_values": ["red", "orange", "yellow", "green", "cyan", "blue", "purple", "white", "pink"],
            "value_meanings": {
                "red": "Red ambient lighting",
                "orange": "Orange ambient lighting (default)",
                "yellow": "Yellow ambient lighting",
                "green": "Green ambient lighting",
                "cyan": "Cyan/turquoise ambient lighting",
                "blue": "Blue ambient lighting",
                "purple": "Purple ambient lighting",
                "white": "White ambient lighting",
                "pink": "Pink ambient lighting",
            },
            "aliases": {
                "红色": "red",
                "橙色": "orange",
                "黄色": "yellow",
                "绿色": "green",
                "青色": "cyan",
                "蓝色": "blue",
                "紫色": "purple",
                "白色": "white",
                "粉色": "pink",
            },
            "default": "orange",
            "notes": "Ambient light must be on to change color. Chinese color names are supported.",
        },
        "reading_light_position": {
            "description": "Target position(s) for reading light control. Supports individual seats or groups.",
            "type": "str",
            "required": True,
            "valid_values": ["driver", "passenger", "rear_left", "rear_right", "front", "rear", "all"],
            "value_meanings": {
                "driver": "Driver seat reading light",
                "passenger": "Front passenger seat reading light",
                "rear_left": "Rear left passenger reading light",
                "rear_right": "Rear right passenger reading light",
                "front": "Both front seat reading lights",
                "rear": "Both rear seat reading lights",
                "all": "All four reading lights",
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
            "notes": "Chinese aliases are supported. Reading light must be on before adjusting brightness.",
        },
        "reading_light_enabled": {
            "description": "Power state of the reading light at the specified position.",
            "type": "bool",
            "required": True,
            "valid_values": [True, False],
            "value_meanings": {
                True: "Turn on reading light - provides focused illumination for reading/tasks",
                False: "Turn off reading light",
            },
            "default": False,
            "notes": "Reading light must be on before brightness can be adjusted.",
        },
        "reading_light_brightness": {
            "description": "Brightness level of the reading light at the specified position.",
            "type": "int",
            "required": True,
            "valid_range": {"min": 1, "max": 10},
            "value_meanings": {
                1: "Minimum brightness",
                5: "Medium brightness",
                10: "Maximum brightness",
            },
            "default": 1,
            "notes": "Reading light must be on to adjust brightness. Higher values provide more illumination for reading.",
        },
    }

    def __init__(self):
        """Initialize with default values."""
        self._lights = {
            "fog_light": {"is_on": False},
            "high_beam": {"is_on": False},
            "low_beam": {"is_on": False, "level": "lowest"},
            "daytime_running": {"is_on": False},
            "left_turn": {"is_on": False},
            "right_turn": {"is_on": False},
            "auto_headlight": {"is_on": False},
            "hazard": {"is_on": False},
            "position": {"is_on": False},
            "tail": {"is_on": False},
            "ambient": {"is_on": False, "color": "orange"},
        }
        self._reading_lights = {
            "driver": self._default_reading_light_state(),
            "passenger": self._default_reading_light_state(),
            "rear_left": self._default_reading_light_state(),
            "rear_right": self._default_reading_light_state(),
        }

    def _default_reading_light_state(self) -> Dict[str, Any]:
        return {"is_on": False, "brightness": 1}

    def _resolve_reading_lights(self, light: str) -> List[str]:
        if not light:
            light = "driver"
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
        if light in mapping:
            return mapping[light]
        raise ValueError("reading light must be driver, passenger, rear_left, rear_right, front, rear, or all")

    def _validate_range(self, value: int, min_value: int, max_value: int, name: str) -> int:
        if not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
        if value < min_value or value > max_value:
            raise ValueError(f"{name} must be between {min_value} and {max_value}")
        return value

    def _normalize_low_beam_level(self, level: str) -> str:
        mapping = {
            "lowest": "lowest",
            "low": "low",
            "medium": "medium",
            "high": "high",
            "highest": "highest",
            "最低": "lowest",
            "低": "low",
            "中": "medium",
            "高": "high",
            "最高": "highest",
        }
        if level in mapping:
            return mapping[level]
        raise ValueError("low_beam level must be lowest, low, medium, high, or highest")

    def _normalize_ambient_color(self, color: str) -> str:
        mapping = {
            "red": "red",
            "orange": "orange",
            "yellow": "yellow",
            "green": "green",
            "cyan": "cyan",
            "blue": "blue",
            "purple": "purple",
            "white": "white",
            "pink": "pink",
            "红色": "red",
            "橙色": "orange",
            "黄色": "yellow",
            "绿色": "green",
            "青色": "cyan",
            "蓝色": "blue",
            "紫色": "purple",
            "白色": "white",
            "粉色": "pink",
        }
        if color in mapping:
            return mapping[color]
        raise ValueError("ambient color must be red, orange, yellow, green, cyan, blue, purple, white, or pink")

    def _set_reading_lights(self, lights: List[str], key: str, value: Any) -> None:
        for light in lights:
            self._reading_lights[light][key] = value

    # === API IMPLEMENTATION METHODS ===
    @api("light")
    def carcontrol_light_set_fog_light(self, enabled: bool) -> Dict[str, Any]:
        """Enable or disable fog light."""
        self._lights["fog_light"]["is_on"] = bool(enabled)
        return {"success": True, "message": "Fog light set successfully", "current_state": self.to_dict()}

    @api("light")
    def carcontrol_light_set_high_beam(self, enabled: bool) -> Dict[str, Any]:
        """Enable or disable high beam."""
        self._lights["high_beam"]["is_on"] = bool(enabled)
        return {"success": True, "message": "High beam set successfully", "current_state": self.to_dict()}

    @api("light")
    def carcontrol_light_set_low_beam_enabled(self, enabled: bool) -> Dict[str, Any]:
        """Enable or disable low beam."""
        self._lights["low_beam"]["is_on"] = bool(enabled)
        return {"success": True, "message": "Low beam set successfully", "current_state": self.to_dict()}

    @api("light")
    def carcontrol_light_set_low_beam_level(self, level: str) -> Dict[str, Any]:
        """Set low beam level (lowest/low/medium/high/highest). Automatically enables low beam if off."""
        try:
            # Ensure low beam is on before setting level
            self._lights["low_beam"]["is_on"] = True
            normalized = self._normalize_low_beam_level(level)
            self._lights["low_beam"]["level"] = normalized
        except ValueError as exc:
            return {"success": False, "message": str(exc), "current_state": self.to_dict()}
        return {"success": True, "message": "Low beam level set successfully", "current_state": self.to_dict()}

    @api("light")
    def carcontrol_light_set_daytime_running(self, enabled: bool) -> Dict[str, Any]:
        """Enable or disable daytime running light."""
        self._lights["daytime_running"]["is_on"] = bool(enabled)
        return {"success": True, "message": "Daytime running light set successfully", "current_state": self.to_dict()}

    @api("light")
    def carcontrol_light_set_left_turn(self, enabled: bool) -> Dict[str, Any]:
        """Enable or disable left turn signal."""
        self._lights["left_turn"]["is_on"] = bool(enabled)
        return {"success": True, "message": "Left turn signal set successfully", "current_state": self.to_dict()}

    @api("light")
    def carcontrol_light_set_right_turn(self, enabled: bool) -> Dict[str, Any]:
        """Enable or disable right turn signal."""
        self._lights["right_turn"]["is_on"] = bool(enabled)
        return {"success": True, "message": "Right turn signal set successfully", "current_state": self.to_dict()}

    @api("light")
    def carcontrol_light_set_auto_headlight(self, enabled: bool) -> Dict[str, Any]:
        """Enable or disable auto headlight."""
        self._lights["auto_headlight"]["is_on"] = bool(enabled)
        return {"success": True, "message": "Auto headlight set successfully", "current_state": self.to_dict()}

    @api("light")
    def carcontrol_light_set_hazard(self, enabled: bool) -> Dict[str, Any]:
        """Enable or disable hazard light."""
        self._lights["hazard"]["is_on"] = bool(enabled)
        return {"success": True, "message": "Hazard light set successfully", "current_state": self.to_dict()}

    @api("light")
    def carcontrol_light_set_position(self, enabled: bool) -> Dict[str, Any]:
        """Enable or disable position light."""
        self._lights["position"]["is_on"] = bool(enabled)
        return {"success": True, "message": "Position light set successfully", "current_state": self.to_dict()}

    @api("light")
    def carcontrol_light_set_tail(self, enabled: bool) -> Dict[str, Any]:
        """Enable or disable tail light."""
        self._lights["tail"]["is_on"] = bool(enabled)
        return {"success": True, "message": "Tail light set successfully", "current_state": self.to_dict()}

    @api("light")
    def carcontrol_light_set_ambient_enabled(self, enabled: bool) -> Dict[str, Any]:
        """Enable or disable ambient light."""
        self._lights["ambient"]["is_on"] = bool(enabled)
        return {"success": True, "message": "Ambient light set successfully", "current_state": self.to_dict()}

    @api("light")
    def carcontrol_light_set_ambient_color(self, color: str) -> Dict[str, Any]:
        """Set ambient light color. Automatically enables ambient light if off."""
        try:
            # Ensure ambient light is on before setting color
            self._lights["ambient"]["is_on"] = True
            normalized = self._normalize_ambient_color(color)
            self._lights["ambient"]["color"] = normalized
        except ValueError as exc:
            return {"success": False, "message": str(exc), "current_state": self.to_dict()}
        return {"success": True, "message": "Ambient light color set successfully", "current_state": self.to_dict()}

    @api("light")
    def carcontrol_light_set_reading_light(self, light: str, enabled: bool) -> Dict[str, Any]:
        """Enable or disable reading light for a position or group."""
        try:
            lights = self._resolve_reading_lights(light)
            self._set_reading_lights(lights, "is_on", bool(enabled))
        except ValueError as exc:
            return {"success": False, "message": str(exc), "current_state": self.to_dict()}
        return {"success": True, "message": "Reading light set successfully", "current_state": self.to_dict()}

    @api("light")
    def carcontrol_light_set_reading_light_brightness(self, light: str, brightness: int) -> Dict[str, Any]:
        """Set reading light brightness (1-10). Automatically enables reading light if off."""
        try:
            brightness = self._validate_range(brightness, 1, 10, "brightness")
            lights = self._resolve_reading_lights(light)
            # Ensure reading lights are on before setting brightness
            self._set_reading_lights(lights, "is_on", True)
            self._set_reading_lights(lights, "brightness", brightness)
        except ValueError as exc:
            return {"success": False, "message": str(exc), "current_state": self.to_dict()}
        return {"success": True, "message": "Reading light brightness set successfully", "current_state": self.to_dict()}

    def to_dict(self) -> Dict[str, Any]:
        """Convert light settings to a dictionary representation."""
        return {
            "fog_light": {
                "value": dict(self._lights["fog_light"]),
                "description": "Fog light settings",
                "type": type(self._lights["fog_light"]).__name__,
            },
            "high_beam": {
                "value": dict(self._lights["high_beam"]),
                "description": "High beam settings",
                "type": type(self._lights["high_beam"]).__name__,
            },
            "low_beam": {
                "value": dict(self._lights["low_beam"]),
                "description": "Low beam settings",
                "type": type(self._lights["low_beam"]).__name__,
            },
            "daytime_running": {
                "value": dict(self._lights["daytime_running"]),
                "description": "Daytime running light settings",
                "type": type(self._lights["daytime_running"]).__name__,
            },
            "left_turn": {
                "value": dict(self._lights["left_turn"]),
                "description": "Left turn signal settings",
                "type": type(self._lights["left_turn"]).__name__,
            },
            "right_turn": {
                "value": dict(self._lights["right_turn"]),
                "description": "Right turn signal settings",
                "type": type(self._lights["right_turn"]).__name__,
            },
            "auto_headlight": {
                "value": dict(self._lights["auto_headlight"]),
                "description": "Auto headlight settings",
                "type": type(self._lights["auto_headlight"]).__name__,
            },
            "hazard": {
                "value": dict(self._lights["hazard"]),
                "description": "Hazard light settings",
                "type": type(self._lights["hazard"]).__name__,
            },
            "position": {
                "value": dict(self._lights["position"]),
                "description": "Position light settings",
                "type": type(self._lights["position"]).__name__,
            },
            "tail": {
                "value": dict(self._lights["tail"]),
                "description": "Tail light settings",
                "type": type(self._lights["tail"]).__name__,
            },
            "ambient": {
                "value": dict(self._lights["ambient"]),
                "description": "Ambient light settings",
                "type": type(self._lights["ambient"]).__name__,
            },
            "reading_light": {
                "value": {
                    "driver": dict(self._reading_lights["driver"]),
                    "passenger": dict(self._reading_lights["passenger"]),
                    "rear_left": dict(self._reading_lights["rear_left"]),
                    "rear_right": dict(self._reading_lights["rear_right"]),
                },
                "description": "Reading light settings",
                "type": type(self._reading_lights["driver"]).__name__,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Light":
        """Create an instance from a dictionary representation."""
        instance = cls()
        instance._lights["fog_light"] = dict(data["fog_light"]["value"])
        instance._lights["high_beam"] = dict(data["high_beam"]["value"])
        instance._lights["low_beam"] = dict(data["low_beam"]["value"])
        instance._lights["daytime_running"] = dict(data["daytime_running"]["value"])
        instance._lights["left_turn"] = dict(data["left_turn"]["value"])
        instance._lights["right_turn"] = dict(data["right_turn"]["value"])
        instance._lights["auto_headlight"] = dict(data["auto_headlight"]["value"])
        instance._lights["hazard"] = dict(data["hazard"]["value"])
        instance._lights["position"] = dict(data["position"]["value"])
        instance._lights["tail"] = dict(data["tail"]["value"])
        instance._lights["ambient"] = dict(data["ambient"]["value"])
        reading = data.get("reading_light", {}).get("value", {})
        if reading:
            instance._reading_lights["driver"] = dict(reading["driver"])
            instance._reading_lights["passenger"] = dict(reading["passenger"])
            instance._reading_lights["rear_left"] = dict(reading["rear_left"])
            instance._reading_lights["rear_right"] = dict(reading["rear_right"])
        return instance
