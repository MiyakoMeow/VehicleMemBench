from ..utils import api
from typing import Dict, Any


class FuelPort:
    """
    Entity class representing the fuel port.
    """

    PARAMS_DESCRIPTION = {
        "open_state": {
            "description": "Open/close state of the fuel port (or charging port for electric vehicles).",
            "type": "bool",
            "required": True,
            "valid_values": [True, False],
            "value_meanings": {
                True: "Open the fuel port - raises the fuel door to allow fueling or charging",
                False: "Close the fuel port - lowers the fuel door back to flush position",
            },
            "default": False,
            "notes": "IMPORTANT: Cannot open the fuel port when it is locked. You must first unlock (set locked to False) before opening. Attempting to open a locked fuel port will result in an error.",
        },
        "locked": {
            "description": "Lock state of the fuel port that prevents unauthorized access.",
            "type": "bool",
            "required": True,
            "valid_values": [True, False],
            "value_meanings": {
                True: "Lock the fuel port - secures the fuel door and prevents it from being opened",
                False: "Unlock the fuel port - allows the fuel door to be opened for fueling or charging",
            },
            "default": True,
            "notes": "Fuel port is locked by default for security. Must be unlocked before opening. Recommended to lock after fueling/charging is complete to prevent theft or tampering.",
        },
    }

    def __init__(self):
        """Initialize with default values."""
        self._is_open = False
        self._is_locked = True

    # === OPEN STATE ===
    @property
    def is_open(self) -> bool:
        """Get fuel port open state."""
        return self._is_open

    @is_open.setter
    def is_open(self, value: bool):
        """Set fuel port open state."""
        if value and self.is_locked:
            raise ValueError("cannot open fuel port when it is locked")
        self._is_open = bool(value)

    # === LOCK STATE ===
    @property
    def is_locked(self) -> bool:
        """Get fuel port lock state."""
        return self._is_locked

    @is_locked.setter
    def is_locked(self, value: bool):
        """Set fuel port lock state."""
        self._is_locked = bool(value)

    # === API IMPLEMENTATION METHODS ===
    @api("fuelPort")
    def carcontrol_fuelPort_set_open(self, open_state: bool) -> Dict[str, Any]:
        """Set fuel port open state."""
        try:
            self.is_open = open_state
        except ValueError as exc:
            return {
                "success": False,
                "message": str(exc),
                "current_state": self.to_dict(),
            }
        return {
            "success": True,
            "message": f"Fuel port {'opened' if open_state else 'closed'} successfully",
            "current_state": self.to_dict(),
        }

    @api("fuelPort")
    def carcontrol_fuelPort_set_locked(self, locked: bool) -> Dict[str, Any]:
        """Set fuel port lock state."""
        self.is_locked = locked
        return {
            "success": True,
            "message": f"Fuel port {'locked' if locked else 'unlocked'} successfully",
            "current_state": self.to_dict(),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert the fuel port to a dictionary representation."""
        return {
            "is_open": {
                "value": self.is_open,
                "description": "Whether the fuel port is open",
                "type": type(self.is_open).__name__,
            },
            "is_locked": {
                "value": self.is_locked,
                "description": "Whether the fuel port is locked",
                "type": type(self.is_locked).__name__,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FuelPort":
        """Create an instance from a dictionary representation."""
        instance = cls()
        instance.is_open = data["is_open"]["value"]
        instance.is_locked = data["is_locked"]["value"]
        return instance
