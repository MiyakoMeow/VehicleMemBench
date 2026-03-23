from ..utils import api
from typing import Dict, Any, List


class Sunshade:
    """
    Entity class representing sunshades with per-position settings.
    """

    PARAMS_DESCRIPTION = {
        "sunshade": {
            "description": "Target sunshade position(s) for control.",
            "type": "str",
            "required": True,
            "valid_values": ["front", "rear", "all"],
            "value_meanings": {
                "front": "Front sunshade only",
                "rear": "Rear sunshade only",
                "all": "Both front and rear sunshades",
            },
            "aliases": {
                "前排": "front",
                "后排": "rear",
                "全部": "all",
            },
            "default": "front",
            "notes": "Chinese aliases are supported.",
        },
        "is_open": {
            "description": "Open/close state of the sunshade.",
            "type": "bool",
            "required": True,
            "valid_values": [True, False],
            "value_meanings": {
                True: "Open the sunshade",
                False: "Close the sunshade",
            },
            "default": False,
            "notes": "Will be auto-opened when setting open_degree to values other than 'close'.",
        },
        "open_degree": {
            "description": "Opening degree/level of the sunshade.",
            "type": "str",
            "required": True,
            "valid_values": ["close", "min", "low", "medium", "high", "max"],
            "value_meanings": {
                "close": "Close the sunshade completely (auto sets is_open to False)",
                "min": "Minimum opening",
                "low": "Low opening",
                "medium": "Medium opening",
                "high": "High opening",
                "max": "Maximum opening (fully open)",
            },
            "aliases": {
                "关闭": "close",
                "最小": "min",
                "较低": "low",
                "中等": "medium",
                "较高": "high",
                "最大": "max",
            },
            "default": "close",
            "notes": "Setting 'close' will auto-close the sunshade. Other values will auto-open the sunshade.",
        },
        "auto_close_on_lock": {
            "description": "Automatically close sunshade when vehicle is locked.",
            "type": "bool",
            "required": True,
            "valid_values": [True, False],
            "value_meanings": {
                True: "Enable auto close - sunshade closes when car is locked",
                False: "Disable auto close",
            },
            "default": False,
            "notes": "Applies to all sunshades when enabled.",
        },
    }

    def __init__(self):
        """Initialize with default values for each sunshade."""
        self._sunshades = {
            "front": self._default_sunshade_state(),
            "rear": self._default_sunshade_state(),
        }

    def _default_sunshade_state(self) -> Dict[str, Any]:
        return {
            "is_open": False,
            "auto_close_on_lock": False,
            "open_degree": "close",
        }

    def _resolve_sunshades(self, sunshade: str) -> List[str]:
        if not sunshade:
            sunshade = "front"
        mapping = {
            "front": ["front"],
            "rear": ["rear"],
            "all": ["front", "rear"],
            "前排": ["front"],
            "后排": ["rear"],
            "全部": ["front", "rear"],
        }
        if sunshade in mapping:
            return mapping[sunshade]
        raise ValueError("sunshade must be front, rear, or all")

    def _normalize_open_degree(self, degree: str) -> str:
        mapping = {
            "close": "close",
            "min": "min",
            "low": "low",
            "medium": "medium",
            "high": "high",
            "max": "max",
            "关闭": "close",
            "最小": "min",
            "较低": "low",
            "中等": "medium",
            "较高": "high",
            "最大": "max",
        }
        if degree in mapping:
            return mapping[degree]
        raise ValueError("open_degree must be close, min, low, medium, high, or max")

    def _set_for_sunshades(self, sunshades: List[str], key: str, value: Any) -> None:
        for sunshade in sunshades:
            self._sunshades[sunshade][key] = value

    def _set_auto_close_for_all(self, enabled: bool) -> None:
        for sunshade in self._sunshades:
            self._sunshades[sunshade]["auto_close_on_lock"] = bool(enabled)

    # === API IMPLEMENTATION METHODS ===
    @api("sunshade")
    def carcontrol_sunshade_set_open(self, sunshade: str, is_open: bool) -> Dict[str, Any]:
        """Set sunshade open state for a position or group."""
        try:
            sunshades = self._resolve_sunshades(sunshade)
            self._set_for_sunshades(sunshades, "is_open", bool(is_open))
        except ValueError as exc:
            return {"success": False, "message": str(exc), "current_state": self.to_dict()}
        return {"success": True, "message": "Sunshade open state set successfully", "current_state": self.to_dict()}

    @api("sunshade")
    def carcontrol_sunshade_set_open_degree(self, sunshade: str, degree: str) -> Dict[str, Any]:
        """Set sunshade open degree (close/min/low/medium/high/max). Auto open/close based on degree."""
        try:
            sunshades = self._resolve_sunshades(sunshade)
            normalized_degree = self._normalize_open_degree(degree)

            if normalized_degree == "close":
                # Auto close when degree is "close"
                self._set_for_sunshades(sunshades, "is_open", False)
                self._set_for_sunshades(sunshades, "open_degree", "close")
            else:
                # Auto open when degree is not "close"
                self._set_for_sunshades(sunshades, "is_open", True)
                self._set_for_sunshades(sunshades, "open_degree", normalized_degree)
        except ValueError as exc:
            return {"success": False, "message": str(exc), "current_state": self.to_dict()}
        return {"success": True, "message": "Sunshade open degree set successfully", "current_state": self.to_dict()}

    @api("sunshade")
    def carcontrol_sunshade_set_auto_close_on_lock(self, sunshade: str, enabled: bool) -> Dict[str, Any]:
        """Enable or disable auto close on lock for all sunshades."""
        try:
            self._resolve_sunshades(sunshade)
            self._set_auto_close_for_all(enabled)
        except ValueError as exc:
            return {"success": False, "message": str(exc), "current_state": self.to_dict()}
        return {
            "success": True,
            "message": "Sunshade auto close on lock set successfully",
            "current_state": self.to_dict(),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert sunshade settings to a dictionary representation."""
        return {
            "front": {
                "value": dict(self._sunshades["front"]),
                "description": "Front sunshade settings",
                "type": type(self._sunshades["front"]).__name__,
            },
            "rear": {
                "value": dict(self._sunshades["rear"]),
                "description": "Rear sunshade settings",
                "type": type(self._sunshades["rear"]).__name__,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Sunshade":
        """Create an instance from a dictionary representation."""
        instance = cls()
        instance._sunshades["front"] = dict(data["front"]["value"])
        instance._sunshades["rear"] = dict(data["rear"]["value"])
        return instance
