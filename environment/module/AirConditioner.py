from ..utils import api
from typing import Dict, Any, List


class AirConditioner:
    """
    Entity class representing air conditioner zones with per-seat settings.
    """

    PARAMS_DESCRIPTION = {
        "zone": {
            "description": "The target zone(s) for air conditioner control. Supports individual seats, grouped zones, or all zones simultaneously.",
            "type": "str",
            "required": True,
            "valid_values": ["driver", "passenger", "rear_left", "rear_right", "front", "rear", "all"],
            "value_meanings": {
                "driver": "Driver seat zone (front left)",
                "passenger": "Front passenger seat zone (front right)",
                "rear_left": "Rear left passenger seat zone",
                "rear_right": "Rear right passenger seat zone",
                "front": "Both front seats (driver + passenger)",
                "rear": "Both rear seats (rear_left + rear_right)",
                "all": "All four zones simultaneously",
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
            "notes": "Chinese aliases are supported for convenience. If not specified, defaults to 'driver' zone.",
        },
        "is_on": {
            "description": "Power state of the air conditioner for the specified zone(s).",
            "type": "bool",
            "required": True,
            "valid_values": [True, False],
            "value_meanings": {
                True: "Turn on the air conditioner",
                False: "Turn off the air conditioner",
            },
            "default": False,
            "notes": "When other settings (temperature, fan_speed, etc.) are modified, the air conditioner will be automatically turned on if it was off.",
        },
        "temperature": {
            "description": "Target temperature setting for the air conditioner in the specified zone(s).",
            "type": "int",
            "required": True,
            "valid_range": {"min": 15, "max": 35},
            "unit": "Celsius (°C)",
            "default": 24,
            "notes": "Must be an integer value. The air conditioner will be automatically turned on if it was off when setting temperature.",
        },
        "fan_speed": {
            "description": "Fan speed level for air circulation intensity in the specified zone(s).",
            "type": "int",
            "required": True,
            "valid_range": {"min": 1, "max": 10},
            "value_meanings": {
                1: "Lowest fan speed (quiet mode)",
                5: "Medium fan speed (balanced)",
                10: "Maximum fan speed (powerful airflow)",
            },
            "default": 5,
            "notes": "Higher values produce stronger airflow but more noise. The air conditioner will be automatically turned on if it was off.",
        },
        "air_direction": {
            "description": "Direction of airflow output from the air conditioner vents in the specified zone(s).",
            "type": "str",
            "required": True,
            "valid_values": ["face", "feet", "window", "face_feet", "face_window", "feet_window", "face_feet_window"],
            "value_meanings": {
                "face": "Airflow directed toward the face/upper body",
                "feet": "Airflow directed toward the feet/floor area",
                "window": "Airflow directed toward the windshield/windows (for defrosting/defogging)",
                "face_feet": "Combined airflow to face and feet",
                "face_window": "Combined airflow to face and window",
                "feet_window": "Combined airflow to feet and window",
                "face_feet_window": "Airflow to all directions (face, feet, and window)",
            },
            "default": "face",
            "notes": "Combined directions allow for more comfortable climate distribution. The air conditioner will be automatically turned on if it was off.",
        },
        "mode": {
            "description": "Operating mode of the air conditioner that determines its primary function.",
            "type": "str",
            "required": True,
            "valid_values": ["auto", "purify", "dehumidify", "defrost", "swing", "defog"],
            "value_meanings": {
                "auto": "Automatic mode - system adjusts settings based on cabin conditions",
                "purify": "Air purification mode - activates air filter to clean cabin air",
                "dehumidify": "Dehumidification mode - removes moisture from cabin air",
                "defrost": "Defrost mode - removes ice/frost from windows (typically uses heat)",
                "swing": "Swing mode - oscillates air direction for even distribution",
                "defog": "Defog mode - clears fog/condensation from windows",
            },
            "default": "auto",
            "notes": "Different modes optimize the AC system for specific conditions. The air conditioner will be automatically turned on if it was off.",
        },
        "circulation": {
            "description": "Air circulation mode that controls the source of air for the climate system.",
            "type": "str",
            "required": True,
            "valid_values": ["inside", "outside"],
            "value_meanings": {
                "inside": "Recirculation mode - recycles cabin air (better for cooling, blocks outside odors/pollution)",
                "outside": "Fresh air mode - draws air from outside the vehicle (better for ventilation, prevents stuffiness)",
            },
            "default": "inside",
            "notes": "Inside circulation is more efficient for cooling but may cause CO2 buildup on long trips. Outside circulation provides fresh air but may let in external pollutants. The air conditioner will be automatically turned on if it was off.",
        },
    }

    def __init__(self):
        """Initialize with default values for each zone."""
        self._zones = {
            "driver": self._default_zone_state(),
            "passenger": self._default_zone_state(),
            "rear_left": self._default_zone_state(),
            "rear_right": self._default_zone_state(),
        }

    def _default_zone_state(self) -> Dict[str, Any]:
        return {
            "is_on": False,
            "temperature": 24,
            "fan_speed": 5,
            "air_direction": "face",
            "mode": "auto",
            "circulation": "inside",
        }

    def _resolve_zones(self, zone: str) -> List[str]:
        if not zone:
            zone = "driver"
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
        if zone in mapping:
            return mapping[zone]
        raise ValueError("zone must be driver, passenger, rear_left, rear_right, front, rear, or all")

    def _validate_range(self, value: int, min_value: int, max_value: int, name: str) -> int:
        if not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
        if value < min_value or value > max_value:
            raise ValueError(f"{name} must be between {min_value} and {max_value}")
        return value

    def _validate_air_direction(self, value: str) -> str:
        valid = {"face", "feet", "window", "face_feet", "face_window", "feet_window", "face_feet_window"}
        if value in valid:
            return value
        raise ValueError("air_direction is invalid")

    def _validate_mode(self, value: str) -> str:
        valid = {"auto", "purify", "dehumidify", "defrost", "swing", "defog"}
        if value in valid:
            return value
        raise ValueError("mode is invalid (auto, purify, dehumidify, defrost, swing, defog)")

    def _ensure_on(self, zones: List[str]) -> None:
        """Auto turn on air conditioner if not already on."""
        for zone in zones:
            if not self._zones[zone]["is_on"]:
                self._zones[zone]["is_on"] = True

    def _validate_circulation(self, value: str) -> str:
        valid = {"inside", "outside"}
        if value in valid:
            return value
        raise ValueError("circulation must be 'inside' or 'outside'")

    def _set_for_zones(self, zones: List[str], key: str, value: Any) -> None:
        for zone in zones:
            self._zones[zone][key] = value

    # === API IMPLEMENTATION METHODS ===
    @api("airConditioner")
    def carcontrol_airConditioner_set_power(self, zone: str, is_on: bool) -> Dict[str, Any]:
        """Set air conditioner power for a zone or group."""
        try:
            zones = self._resolve_zones(zone)
            self._set_for_zones(zones, "is_on", bool(is_on))
        except ValueError as exc:
            return {"success": False, "message": str(exc), "current_state": self.to_dict()}
        return {"success": True, "message": "Air conditioner power set successfully", "current_state": self.to_dict()}

    @api("airConditioner")
    def carcontrol_airConditioner_set_temperature(self, zone: str, temperature: int) -> Dict[str, Any]:
        """Set air conditioner temperature (15-35)."""
        try:
            temperature = self._validate_range(temperature, 15, 35, "temperature")
            zones = self._resolve_zones(zone)
            self._ensure_on(zones)
            self._set_for_zones(zones, "temperature", temperature)
        except ValueError as exc:
            return {"success": False, "message": str(exc), "current_state": self.to_dict()}
        return {"success": True, "message": "Air conditioner temperature set successfully", "current_state": self.to_dict()}

    @api("airConditioner")
    def carcontrol_airConditioner_set_fan_speed(self, zone: str, speed: int) -> Dict[str, Any]:
        """Set air conditioner fan speed (1-10)."""
        try:
            speed = self._validate_range(speed, 1, 10, "fan_speed")
            zones = self._resolve_zones(zone)
            self._ensure_on(zones)
            self._set_for_zones(zones, "fan_speed", speed)
        except ValueError as exc:
            return {"success": False, "message": str(exc), "current_state": self.to_dict()}
        return {"success": True, "message": "Air conditioner fan speed set successfully", "current_state": self.to_dict()}

    @api("airConditioner")
    def carcontrol_airConditioner_set_air_direction(self, zone: str, direction: str) -> Dict[str, Any]:
        """Set air conditioner air direction."""
        try:
            direction = self._validate_air_direction(direction)
            zones = self._resolve_zones(zone)
            self._ensure_on(zones)
            self._set_for_zones(zones, "air_direction", direction)
        except ValueError as exc:
            return {"success": False, "message": str(exc), "current_state": self.to_dict()}
        return {"success": True, "message": "Air conditioner direction set successfully", "current_state": self.to_dict()}

    @api("airConditioner")
    def carcontrol_airConditioner_set_mode(self, zone: str, mode: str) -> Dict[str, Any]:
        """Set air conditioner mode (auto/purify/dehumidify/defrost/swing/defog)."""
        try:
            mode = self._validate_mode(mode)
            zones = self._resolve_zones(zone)
            self._ensure_on(zones)
            self._set_for_zones(zones, "mode", mode)
        except ValueError as exc:
            return {"success": False, "message": str(exc), "current_state": self.to_dict()}
        return {"success": True, "message": "Air conditioner mode set successfully", "current_state": self.to_dict()}

    @api("airConditioner")
    def carcontrol_airConditioner_set_circulation(self, zone: str, circulation: str) -> Dict[str, Any]:
        """Set air conditioner circulation (inside/outside)."""
        try:
            circulation = self._validate_circulation(circulation)
            zones = self._resolve_zones(zone)
            self._ensure_on(zones)
            self._set_for_zones(zones, "circulation", circulation)
        except ValueError as exc:
            return {"success": False, "message": str(exc), "current_state": self.to_dict()}
        return {"success": True, "message": "Air conditioner circulation set successfully", "current_state": self.to_dict()}

    def to_dict(self) -> Dict[str, Any]:
        """Convert air conditioner settings to a dictionary representation."""
        return {
            "driver": {
                "value": dict(self._zones["driver"]),
                "description": "Driver zone settings",
                "type": type(self._zones["driver"]).__name__,
            },
            "passenger": {
                "value": dict(self._zones["passenger"]),
                "description": "Passenger zone settings",
                "type": type(self._zones["passenger"]).__name__,
            },
            "rear_left": {
                "value": dict(self._zones["rear_left"]),
                "description": "Rear left zone settings",
                "type": type(self._zones["rear_left"]).__name__,
            },
            "rear_right": {
                "value": dict(self._zones["rear_right"]),
                "description": "Rear right zone settings",
                "type": type(self._zones["rear_right"]).__name__,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AirConditioner":
        """Create an instance from a dictionary representation."""
        instance = cls()
        instance._zones["driver"] = dict(data["driver"]["value"])
        instance._zones["passenger"] = dict(data["passenger"]["value"])
        instance._zones["rear_left"] = dict(data["rear_left"]["value"])
        instance._zones["rear_right"] = dict(data["rear_right"]["value"])
        return instance
