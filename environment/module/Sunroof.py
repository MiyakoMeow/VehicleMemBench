from ..utils import api
from typing import Dict, Any


class Sunroof:
    """
    Entity class representing the sunroof.
    """

    PARAMS_DESCRIPTION = {
        "is_locked": {
            "description": "Lock state of the sunroof to prevent opening.",
            "type": "bool",
            "required": True,
            "valid_values": [True, False],
            "value_meanings": {
                True: "Lock the sunroof - prevents opening",
                False: "Unlock the sunroof - allows opening",
            },
            "default": False,
            "notes": "Will be auto-unlocked when setting open_degree > 0, and auto-locked when setting open_degree = 0.",
        },
        "open_degree": {
            "description": "Opening degree/percentage of the sunroof.",
            "type": "int",
            "required": True,
            "valid_range": {"min": 0, "max": 100},
            "value_meanings": {
                0: "Fully closed (will auto-lock)",
                50: "Half open",
                100: "Fully open",
            },
            "default": 0,
            "notes": "Setting degree > 0 will auto-unlock, setting degree = 0 will auto-lock.",
        },
    }

    def __init__(self):
        """Initialize with default values."""
        self._is_locked = False
        self._open_degree = 0

    # === LOCK STATE ===
    @property
    def is_locked(self) -> bool:
        """Get sunroof lock state."""
        return self._is_locked

    @is_locked.setter
    def is_locked(self, value: bool):
        """Set sunroof lock state."""
        self._is_locked = bool(value)

    # === OPEN DEGREE ===
    @property
    def open_degree(self) -> int:
        """Get open degree (0-100)."""
        return self._open_degree

    @open_degree.setter
    def open_degree(self, value: int):
        """Set open degree (0-100). Auto unlock when > 0, auto lock when = 0."""
        self._open_degree = self._validate_open_degree(value)

    def _validate_open_degree(self, value: int) -> int:
        if not isinstance(value, int):
            raise ValueError("open_degree must be an integer")
        if value < 0 or value > 100:
            raise ValueError("open_degree must be between 0 and 100")
        return value

    # === API IMPLEMENTATION METHODS ===
    @api("sunroof")
    def carcontrol_sunroof_set_locked(self, locked: bool) -> Dict[str, Any]:
        """Set sunroof lock state."""
        self.is_locked = locked
        return {
            "success": True,
            "message": f"Sunroof {'locked' if locked else 'unlocked'} successfully",
            "current_state": self.to_dict(),
        }

    @api("sunroof")
    def carcontrol_sunroof_set_open_degree(self, degree: int) -> Dict[str, Any]:
        """Set sunroof open degree (0-100). Auto unlock when > 0, auto lock when = 0."""
        try:
            # Auto unlock when opening, auto lock when closing
            if degree > 0:
                self._is_locked = False
            self.open_degree = degree
            if degree == 0:
                self._is_locked = True
        except ValueError as exc:
            return {
                "success": False,
                "message": str(exc),
                "current_state": self.to_dict(),
            }
        return {
            "success": True,
            "message": "Sunroof open degree set successfully",
            "current_state": self.to_dict(),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert the sunroof to a dictionary representation."""
        return {
            "is_locked": {
                "value": self.is_locked,
                "description": "Whether the sunroof is locked",
                "type": type(self.is_locked).__name__,
            },
            "open_degree": {
                "value": self.open_degree,
                "description": "Sunroof open degree (0-100)",
                "type": type(self.open_degree).__name__,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Sunroof":
        """Create an instance from a dictionary representation."""
        instance = cls()
        instance.is_locked = data["is_locked"]["value"]
        instance.open_degree = data["open_degree"]["value"]
        return instance
