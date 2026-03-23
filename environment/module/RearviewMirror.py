from ..utils import api
from typing import Dict, Any, List


class RearviewMirror:
    """
    Entity class representing rearview mirrors with per-side settings.
    """

    PARAMS_DESCRIPTION = {
        "side": {
            "description": "Target side(s) for rearview mirror control.",
            "type": "str",
            "required": True,
            "valid_values": ["left", "right", "both"],
            "value_meanings": {
                "left": "Left side rearview mirror only",
                "right": "Right side rearview mirror only",
                "both": "Both left and right rearview mirrors simultaneously",
            },
            "aliases": {
                "左": "left",
                "右": "right",
                "两侧": "both",
            },
            "default": "both",
            "notes": "Chinese aliases are supported.",
        },
        "is_on": {
            "description": "Power state of the rearview mirror electric adjustment system.",
            "type": "bool",
            "required": True,
            "valid_values": [True, False],
            "value_meanings": {
                True: "Turn on mirror power - enables electric adjustment",
                False: "Turn off mirror power",
            },
            "default": False,
            "notes": "All other mirror operations will automatically turn on the mirror if it was off.",
        },
        "height_position": {
            "description": "Vertical position of the rearview mirror (up/down adjustment).",
            "type": "int",
            "required": True,
            "valid_range": {"min": 0, "max": 100},
            "value_meanings": {
                0: "Lowest position",
                50: "Middle position",
                100: "Highest position",
            },
            "default": 0,
            "notes": "System will auto turn on if needed. Adjust for optimal rear visibility based on driver height.",
        },
        "horizontal_position": {
            "description": "Horizontal position of the rearview mirror (left/right adjustment).",
            "type": "int",
            "required": True,
            "valid_range": {"min": 0, "max": 100},
            "value_meanings": {
                0: "Most inward position",
                50: "Middle position",
                100: "Most outward position",
            },
            "default": 0,
            "notes": "System will auto turn on if needed. Adjust to minimize blind spots.",
        },
        "auto_reverse_tilt": {
            "description": "Automatically tilt mirror downward when reversing to see curb/obstacles.",
            "type": "bool",
            "required": True,
            "valid_values": [True, False],
            "value_meanings": {
                True: "Enable auto reverse tilt - mirror tilts down when gear is in reverse",
                False: "Disable auto reverse tilt",
            },
            "default": False,
            "notes": "System will auto turn on if needed. Useful for parking assistance.",
        },
        "auto_fold_on_lock": {
            "description": "Automatically fold mirrors when vehicle is locked.",
            "type": "bool",
            "required": True,
            "valid_values": [True, False],
            "value_meanings": {
                True: "Enable auto fold - mirrors fold when car is locked",
                False: "Disable auto fold",
            },
            "default": False,
            "notes": "System will auto turn on if needed. Protects mirrors in tight parking spaces.",
        },
        "heating_enabled": {
            "description": "Mirror heating to remove fog, frost, or ice from mirror surface.",
            "type": "bool",
            "required": True,
            "valid_values": [True, False],
            "value_meanings": {
                True: "Enable mirror heating - clears condensation and ice",
                False: "Disable mirror heating",
            },
            "default": False,
            "notes": "System will auto turn on if needed. Recommended in cold or humid conditions.",
        },
    }

    def __init__(self):
        """Initialize with default values for left and right mirrors."""
        self._mirrors = {
            "left": {
                "is_on": False,
                "height_position": 0,
                "horizontal_position": 0,
                "auto_reverse_tilt": False,
                "auto_fold_on_lock": False,
                "heating_enabled": False,
            },
            "right": {
                "is_on": False,
                "height_position": 0,
                "horizontal_position": 0,
                "auto_reverse_tilt": False,
                "auto_fold_on_lock": False,
                "heating_enabled": False,
            },
        }

    def _resolve_sides(self, side: str) -> List[str]:
        if side == "both":
            return ["left", "right"]
        if side in ("left", "right"):
            return [side]
        raise ValueError("side must be 'left', 'right', or 'both'")

    def _ensure_on(self, sides: List[str]) -> None:
        """Auto turn on mirror power if not already on."""
        for s in sides:
            if not self._mirrors[s]["is_on"]:
                self._mirrors[s]["is_on"] = True

    def _validate_position(self, value: int, name: str) -> int:
        if not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
        if value < 0 or value > 100:
            raise ValueError(f"{name} must be between 0 and 100")
        return value

    # === API IMPLEMENTATION METHODS ===
    @api("rearviewMirror")
    def carcontrol_rearviewMirror_set_power(self, side: str, is_on: bool) -> Dict[str, Any]:
        """Set mirror power for left, right, or both."""
        try:
            for s in self._resolve_sides(side):
                self._mirrors[s]["is_on"] = bool(is_on)
        except ValueError as exc:
            return {"success": False, "message": str(exc), "current_state": self.to_dict()}
        return {"success": True, "message": "Mirror power set successfully", "current_state": self.to_dict()}

    @api("rearviewMirror")
    def carcontrol_rearviewMirror_set_height_position(self, side: str, value: int) -> Dict[str, Any]:
        """Set mirror height position (0-100) for left, right, or both."""
        try:
            value = self._validate_position(value, "height_position")
            sides = self._resolve_sides(side)
            self._ensure_on(sides)
            for s in sides:
                self._mirrors[s]["height_position"] = value
        except ValueError as exc:
            return {"success": False, "message": str(exc), "current_state": self.to_dict()}
        return {"success": True, "message": "Height position set successfully", "current_state": self.to_dict()}

    @api("rearviewMirror")
    def carcontrol_rearviewMirror_set_horizontal_position(self, side: str, value: int) -> Dict[str, Any]:
        """Set mirror horizontal position (0-100) for left, right, or both."""
        try:
            value = self._validate_position(value, "horizontal_position")
            sides = self._resolve_sides(side)
            self._ensure_on(sides)
            for s in sides:
                self._mirrors[s]["horizontal_position"] = value
        except ValueError as exc:
            return {"success": False, "message": str(exc), "current_state": self.to_dict()}
        return {"success": True, "message": "Horizontal position set successfully", "current_state": self.to_dict()}

    @api("rearviewMirror")
    def carcontrol_rearviewMirror_set_auto_reverse_tilt(self, side: str, enabled: bool) -> Dict[str, Any]:
        """Set auto reverse tilt for left, right, or both."""
        try:
            sides = self._resolve_sides(side)
            self._ensure_on(sides)
            for s in sides:
                self._mirrors[s]["auto_reverse_tilt"] = bool(enabled)
        except ValueError as exc:
            return {"success": False, "message": str(exc), "current_state": self.to_dict()}
        return {"success": True, "message": "Auto reverse tilt set successfully", "current_state": self.to_dict()}

    @api("rearviewMirror")
    def carcontrol_rearviewMirror_set_auto_fold_on_lock(self, side: str, enabled: bool) -> Dict[str, Any]:
        """Set auto fold on lock (fold mirrors automatically when the car is locked) for left, right, or both."""
        try:
            sides = self._resolve_sides(side)
            self._ensure_on(sides)
            for s in sides:
                self._mirrors[s]["auto_fold_on_lock"] = bool(enabled)
        except ValueError as exc:
            return {"success": False, "message": str(exc), "current_state": self.to_dict()}
        return {"success": True, "message": "Auto fold on lock set successfully", "current_state": self.to_dict()}

    @api("rearviewMirror")
    def carcontrol_rearviewMirror_set_heating_enabled(self, side: str, enabled: bool) -> Dict[str, Any]:
        """Set mirror heating for left, right, or both."""
        try:
            sides = self._resolve_sides(side)
            self._ensure_on(sides)
            for s in sides:
                self._mirrors[s]["heating_enabled"] = bool(enabled)
        except ValueError as exc:
            return {"success": False, "message": str(exc), "current_state": self.to_dict()}
        return {"success": True, "message": "Mirror heating set successfully", "current_state": self.to_dict()}

    def to_dict(self) -> Dict[str, Any]:
        """Convert mirrors to a dictionary representation."""
        return {
            "left": {
                "value": dict(self._mirrors["left"]),
                "description": "Left rearview mirror settings",
                "type": type(self._mirrors["left"]).__name__,
            },
            "right": {
                "value": dict(self._mirrors["right"]),
                "description": "Right rearview mirror settings",
                "type": type(self._mirrors["right"]).__name__,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RearviewMirror":
        """Create an instance from a dictionary representation."""
        instance = cls()
        instance._mirrors["left"] = dict(data["left"]["value"])
        instance._mirrors["right"] = dict(data["right"]["value"])
        return instance
