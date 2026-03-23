from ..utils import api
from typing import Dict, Any, List


class Seat:
    """
    Entity class representing vehicle seats with per-position settings.
    """

    PARAMS_DESCRIPTION = {
        "seat": {
            "description": "Target seat position(s) for control.",
            "type": "str",
            "required": True,
            "valid_values": ["driver", "passenger", "rear_left", "rear_right", "front", "rear", "all"],
            "value_meanings": {
                "driver": "Driver seat only",
                "passenger": "Front passenger seat only",
                "rear_left": "Rear left passenger seat only",
                "rear_right": "Rear right passenger seat only",
                "front": "Both front seats (driver and passenger)",
                "rear": "Both rear seats (rear_left and rear_right)",
                "all": "All four seats",
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
            "notes": "Chinese aliases are supported.",
        },
        "heating_mode": {
            "description": "Seat heating function to warm the seat surface.",
            "type": "bool",
            "required": True,
            "valid_values": [True, False],
            "value_meanings": {
                True: "Enable seat heating",
                False: "Disable seat heating",
            },
            "default": False,
            "notes": "Recommended in cold weather for comfort.",
        },
        "heating_level": {
            "description": "Intensity level of seat heating.",
            "type": "int",
            "required": True,
            "valid_range": {"min": 1, "max": 3},
            "value_meanings": {
                1: "Low heat",
                2: "Medium heat",
                3: "High heat",
            },
            "default": 1,
            "notes": "Higher levels provide more warmth but consume more power.",
        },
        "massage_mode": {
            "description": "Seat massage function for comfort during long drives.",
            "type": "bool",
            "required": True,
            "valid_values": [True, False],
            "value_meanings": {
                True: "Enable seat massage",
                False: "Disable seat massage",
            },
            "default": False,
            "notes": "Helps reduce fatigue on long journeys.",
        },
        "massage_level": {
            "description": "Intensity level of seat massage.",
            "type": "int",
            "required": True,
            "valid_range": {"min": 1, "max": 3},
            "value_meanings": {
                1: "Gentle massage",
                2: "Medium massage",
                3: "Strong massage",
            },
            "default": 1,
            "notes": "Adjust based on personal preference.",
        },
        "ventilation_enabled": {
            "description": "Seat ventilation/cooling function to circulate air through the seat.",
            "type": "bool",
            "required": True,
            "valid_values": [True, False],
            "value_meanings": {
                True: "Enable seat ventilation",
                False: "Disable seat ventilation",
            },
            "default": False,
            "notes": "Recommended in hot weather to reduce sweating.",
        },
        "ventilation_speed": {
            "description": "Fan speed for seat ventilation.",
            "type": "int",
            "required": True,
            "valid_range": {"min": 1, "max": 5},
            "value_meanings": {
                1: "Lowest fan speed",
                2: "Low fan speed",
                3: "Medium fan speed",
                4: "High fan speed",
                5: "Maximum fan speed",
            },
            "default": 1,
            "notes": "Higher speeds provide more cooling but may be noisier.",
        },
        "horizontal_position": {
            "description": "Front-to-back position of the seat (distance from steering wheel/dashboard).",
            "type": "int",
            "required": True,
            "valid_range": {"min": 0, "max": 100},
            "value_meanings": {
                0: "Most forward position (closest to dashboard)",
                50: "Middle position (default)",
                100: "Most rearward position (farthest from dashboard)",
            },
            "default": 50,
            "notes": "Adjust for comfortable reach to pedals and steering wheel.",
        },
        "vertical_position": {
            "description": "Height position of the seat.",
            "type": "int",
            "required": True,
            "valid_range": {"min": 0, "max": 100},
            "value_meanings": {
                0: "Lowest position",
                50: "Middle position (default)",
                100: "Highest position",
            },
            "default": 50,
            "notes": "Adjust for optimal visibility over the dashboard and hood.",
        },
        "folded": {
            "description": "Seat folding state (primarily for rear seats to expand cargo space).",
            "type": "bool",
            "required": True,
            "valid_values": [True, False],
            "value_meanings": {
                True: "Fold the seat down",
                False: "Unfold the seat (normal position)",
            },
            "default": False,
            "notes": "Folding rear seats increases trunk/cargo capacity.",
        },
        "cushion_extension": {
            "description": "Extension length of the seat cushion (thigh support).",
            "type": "int",
            "required": True,
            "valid_range": {"min": 0, "max": 100},
            "value_meanings": {
                0: "Minimum extension (shortest cushion)",
                50: "Medium extension (default)",
                100: "Maximum extension (longest cushion)",
            },
            "default": 50,
            "notes": "Extend for better thigh support, especially for taller occupants.",
        },
        "cushion_angle": {
            "description": "Tilt angle of the seat cushion.",
            "type": "int",
            "required": True,
            "valid_range": {"min": 0, "max": 100},
            "value_meanings": {
                0: "Most tilted down (front lower)",
                50: "Level position (default)",
                100: "Most tilted up (front higher)",
            },
            "default": 50,
            "notes": "Adjust for comfortable thigh angle and posture.",
        },
        "backrest_angle": {
            "description": "Recline angle of the seat backrest.",
            "type": "int",
            "required": True,
            "valid_range": {"min": 0, "max": 100},
            "value_meanings": {
                0: "Most upright position",
                50: "Slightly reclined (default)",
                100: "Most reclined position",
            },
            "default": 50,
            "notes": "More recline for relaxation, more upright for alertness while driving.",
        },
        "leg_support_height": {
            "description": "Height of the leg rest/support (for seats with leg rests).",
            "type": "int",
            "required": True,
            "valid_range": {"min": 0, "max": 100},
            "value_meanings": {
                0: "Leg support fully lowered/retracted",
                50: "Medium height (default)",
                100: "Leg support fully raised/extended",
            },
            "default": 50,
            "notes": "Primarily for rear seats or luxury front seats with leg rest feature.",
        },
        "foot_support_height": {
            "description": "Height of the foot rest/support.",
            "type": "int",
            "required": True,
            "valid_range": {"min": 0, "max": 100},
            "value_meanings": {
                0: "Foot support fully lowered/retracted",
                50: "Medium height (default)",
                100: "Foot support fully raised/extended",
            },
            "default": 50,
            "notes": "Primarily for rear seats with foot rest feature for relaxation.",
        },
        "headrest_height": {
            "description": "Height of the headrest.",
            "type": "int",
            "required": True,
            "valid_range": {"min": 0, "max": 100},
            "value_meanings": {
                0: "Lowest headrest position",
                50: "Medium height (default)",
                100: "Highest headrest position",
            },
            "default": 50,
            "notes": "Adjust so the headrest center aligns with the back of your head for safety.",
        },
    }

    def __init__(self):
        """Initialize with default values for each seat position."""
        self._seats = {
            "driver": self._default_seat_state(),
            "passenger": self._default_seat_state(),
            "rear_left": self._default_seat_state(),
            "rear_right": self._default_seat_state(),
        }

    def _default_seat_state(self) -> Dict[str, Any]:
        return {
            "heating_mode": False,
            "heating_level": 1,
            "massage_mode": False,
            "massage_level": 1,
            "ventilation_enabled": False,
            "ventilation_speed": 1,
            "horizontal_position": 50,
            "vertical_position": 50,
            "folded": False,
            "cushion_extension": 50,
            "cushion_angle": 50,
            "backrest_angle": 50,
            "leg_support_height": 50,
            "foot_support_height": 50,
            "headrest_height": 50,
        }

    def _resolve_seats(self, seat: str) -> List[str]:
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
        if seat in mapping:
            return mapping[seat]
        raise ValueError("seat must be driver, passenger, rear_left, rear_right, front, rear, or all")

    def _validate_range(self, value: int, min_value: int, max_value: int, name: str) -> int:
        if not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
        if value < min_value or value > max_value:
            raise ValueError(f"{name} must be between {min_value} and {max_value}")
        return value

    def _set_for_seats(self, seats: List[str], key: str, value: Any) -> None:
        for seat in seats:
            self._seats[seat][key] = value

    # === API IMPLEMENTATION METHODS ===
    @api("seat")
    def carcontrol_seat_set_heating_mode(self, seat: str, enabled: bool) -> Dict[str, Any]:
        """Enable or disable seat heating."""
        try:
            seats = self._resolve_seats(seat)
            self._set_for_seats(seats, "heating_mode", bool(enabled))
        except ValueError as exc:
            return {"success": False, "message": str(exc), "current_state": self.to_dict()}
        return {"success": True, "message": "Seat heating mode set successfully", "current_state": self.to_dict()}

    @api("seat")
    def carcontrol_seat_set_heating_level(self, seat: str, level: int) -> Dict[str, Any]:
        """Set seat heating level (1-3). Automatically enables heating if off."""
        try:
            level = self._validate_range(level, 1, 3, "heating_level")
            seats = self._resolve_seats(seat)
            # Ensure heating is enabled before setting level
            self._set_for_seats(seats, "heating_mode", True)
            self._set_for_seats(seats, "heating_level", level)
        except ValueError as exc:
            return {"success": False, "message": str(exc), "current_state": self.to_dict()}
        return {"success": True, "message": "Seat heating level set successfully", "current_state": self.to_dict()}

    @api("seat")
    def carcontrol_seat_set_massage_mode(self, seat: str, enabled: bool) -> Dict[str, Any]:
        """Enable or disable seat massage."""
        try:
            seats = self._resolve_seats(seat)
            self._set_for_seats(seats, "massage_mode", bool(enabled))
        except ValueError as exc:
            return {"success": False, "message": str(exc), "current_state": self.to_dict()}
        return {"success": True, "message": "Seat massage mode set successfully", "current_state": self.to_dict()}

    @api("seat")
    def carcontrol_seat_set_massage_level(self, seat: str, level: int) -> Dict[str, Any]:
        """Set seat massage level (1-3). Automatically enables massage if off."""
        try:
            level = self._validate_range(level, 1, 3, "massage_level")
            seats = self._resolve_seats(seat)
            # Ensure massage is enabled before setting level
            self._set_for_seats(seats, "massage_mode", True)
            self._set_for_seats(seats, "massage_level", level)
        except ValueError as exc:
            return {"success": False, "message": str(exc), "current_state": self.to_dict()}
        return {"success": True, "message": "Seat massage level set successfully", "current_state": self.to_dict()}

    @api("seat")
    def carcontrol_seat_set_ventilation_enabled(self, seat: str, enabled: bool) -> Dict[str, Any]:
        """Enable or disable seat ventilation."""
        try:
            seats = self._resolve_seats(seat)
            self._set_for_seats(seats, "ventilation_enabled", bool(enabled))
        except ValueError as exc:
            return {"success": False, "message": str(exc), "current_state": self.to_dict()}
        return {"success": True, "message": "Seat ventilation set successfully", "current_state": self.to_dict()}

    @api("seat")
    def carcontrol_seat_set_ventilation_speed(self, seat: str, speed: int) -> Dict[str, Any]:
        """Set seat ventilation speed (1-5). Automatically enables ventilation if off."""
        try:
            speed = self._validate_range(speed, 1, 5, "ventilation_speed")
            seats = self._resolve_seats(seat)
            # Ensure ventilation is enabled before setting speed
            self._set_for_seats(seats, "ventilation_enabled", True)
            self._set_for_seats(seats, "ventilation_speed", speed)
        except ValueError as exc:
            return {"success": False, "message": str(exc), "current_state": self.to_dict()}
        return {"success": True, "message": "Seat ventilation speed set successfully", "current_state": self.to_dict()}

    @api("seat")
    def carcontrol_seat_set_horizontal_position(self, seat: str, value: int) -> Dict[str, Any]:
        """Set seat horizontal position (0-100)."""
        try:
            value = self._validate_range(value, 0, 100, "horizontal_position")
            seats = self._resolve_seats(seat)
            self._set_for_seats(seats, "horizontal_position", value)
        except ValueError as exc:
            return {"success": False, "message": str(exc), "current_state": self.to_dict()}
        return {"success": True, "message": "Seat horizontal position set successfully", "current_state": self.to_dict()}

    @api("seat")
    def carcontrol_seat_set_vertical_position(self, seat: str, value: int) -> Dict[str, Any]:
        """Set seat vertical position (0-100)."""
        try:
            value = self._validate_range(value, 0, 100, "vertical_position")
            seats = self._resolve_seats(seat)
            self._set_for_seats(seats, "vertical_position", value)
        except ValueError as exc:
            return {"success": False, "message": str(exc), "current_state": self.to_dict()}
        return {"success": True, "message": "Seat vertical position set successfully", "current_state": self.to_dict()}

    @api("seat")
    def carcontrol_seat_set_folded(self, seat: str, folded: bool) -> Dict[str, Any]:
        """Fold or unfold the seat."""
        try:
            seats = self._resolve_seats(seat)
            self._set_for_seats(seats, "folded", bool(folded))
        except ValueError as exc:
            return {"success": False, "message": str(exc), "current_state": self.to_dict()}
        return {"success": True, "message": "Seat folded state set successfully", "current_state": self.to_dict()}

    @api("seat")
    def carcontrol_seat_set_cushion_extension(self, seat: str, value: int) -> Dict[str, Any]:
        """Set seat cushion extension (0-100)."""
        try:
            value = self._validate_range(value, 0, 100, "cushion_extension")
            seats = self._resolve_seats(seat)
            self._set_for_seats(seats, "cushion_extension", value)
        except ValueError as exc:
            return {"success": False, "message": str(exc), "current_state": self.to_dict()}
        return {"success": True, "message": "Seat cushion extension set successfully", "current_state": self.to_dict()}

    @api("seat")
    def carcontrol_seat_set_cushion_angle(self, seat: str, value: int) -> Dict[str, Any]:
        """Set seat cushion angle (0-100)."""
        try:
            value = self._validate_range(value, 0, 100, "cushion_angle")
            seats = self._resolve_seats(seat)
            self._set_for_seats(seats, "cushion_angle", value)
        except ValueError as exc:
            return {"success": False, "message": str(exc), "current_state": self.to_dict()}
        return {"success": True, "message": "Seat cushion angle set successfully", "current_state": self.to_dict()}

    @api("seat")
    def carcontrol_seat_set_backrest_angle(self, seat: str, value: int) -> Dict[str, Any]:
        """Set seat backrest angle (0-100)."""
        try:
            value = self._validate_range(value, 0, 100, "backrest_angle")
            seats = self._resolve_seats(seat)
            self._set_for_seats(seats, "backrest_angle", value)
        except ValueError as exc:
            return {"success": False, "message": str(exc), "current_state": self.to_dict()}
        return {"success": True, "message": "Seat backrest angle set successfully", "current_state": self.to_dict()}

    @api("seat")
    def carcontrol_seat_set_leg_support_height(self, seat: str, value: int) -> Dict[str, Any]:
        """Set leg support height (0-100)."""
        try:
            value = self._validate_range(value, 0, 100, "leg_support_height")
            seats = self._resolve_seats(seat)
            self._set_for_seats(seats, "leg_support_height", value)
        except ValueError as exc:
            return {"success": False, "message": str(exc), "current_state": self.to_dict()}
        return {"success": True, "message": "Seat leg support height set successfully", "current_state": self.to_dict()}

    @api("seat")
    def carcontrol_seat_set_foot_support_height(self, seat: str, value: int) -> Dict[str, Any]:
        """Set foot support height (0-100)."""
        try:
            value = self._validate_range(value, 0, 100, "foot_support_height")
            seats = self._resolve_seats(seat)
            self._set_for_seats(seats, "foot_support_height", value)
        except ValueError as exc:
            return {"success": False, "message": str(exc), "current_state": self.to_dict()}
        return {"success": True, "message": "Seat foot support height set successfully", "current_state": self.to_dict()}

    @api("seat")
    def carcontrol_seat_set_headrest_height(self, seat: str, value: int) -> Dict[str, Any]:
        """Set headrest height (0-100)."""
        try:
            value = self._validate_range(value, 0, 100, "headrest_height")
            seats = self._resolve_seats(seat)
            self._set_for_seats(seats, "headrest_height", value)
        except ValueError as exc:
            return {"success": False, "message": str(exc), "current_state": self.to_dict()}
        return {"success": True, "message": "Seat headrest height set successfully", "current_state": self.to_dict()}

    def to_dict(self) -> Dict[str, Any]:
        """Convert seat settings to a dictionary representation."""
        return {
            "driver": {
                "value": dict(self._seats["driver"]),
                "description": "Driver seat settings",
                "type": type(self._seats["driver"]).__name__,
            },
            "passenger": {
                "value": dict(self._seats["passenger"]),
                "description": "Passenger seat settings",
                "type": type(self._seats["passenger"]).__name__,
            },
            "rear_left": {
                "value": dict(self._seats["rear_left"]),
                "description": "Rear left seat settings",
                "type": type(self._seats["rear_left"]).__name__,
            },
            "rear_right": {
                "value": dict(self._seats["rear_right"]),
                "description": "Rear right seat settings",
                "type": type(self._seats["rear_right"]).__name__,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Seat":
        """Create an instance from a dictionary representation."""
        instance = cls()
        instance._seats["driver"] = dict(data["driver"]["value"])
        instance._seats["passenger"] = dict(data["passenger"]["value"])
        instance._seats["rear_left"] = dict(data["rear_left"]["value"])
        instance._seats["rear_right"] = dict(data["rear_right"]["value"])
        return instance
