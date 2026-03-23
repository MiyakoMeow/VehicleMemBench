from ..utils import api
from typing import Dict, Any


class Bluetooth:
    """
    Entity class representing Bluetooth connection state.
    """

    PARAMS_DESCRIPTION = {
        "connected": {
            "description": "Bluetooth connection state that controls whether the vehicle's Bluetooth module is connected to an external device.",
            "type": "bool",
            "required": True,
            "valid_values": [True, False],
            "value_meanings": {
                True: "Connect Bluetooth - enables pairing and communication with external devices (phones, audio players, etc.)",
                False: "Disconnect Bluetooth - terminates the current Bluetooth connection",
            },
            "default": False,
            "notes": "When connected, the vehicle can receive audio streams, phone calls, and other data from paired devices. Disconnecting will stop all Bluetooth-related functions.",
        },
    }

    def __init__(self):
        """Initialize with default values."""
        self._is_connected = False

    # === CONNECTION STATE ===
    @property
    def is_connected(self) -> bool:
        """Get Bluetooth connection state."""
        return self._is_connected

    @is_connected.setter
    def is_connected(self, value: bool):
        """Set Bluetooth connection state."""
        self._is_connected = bool(value)

    # === API IMPLEMENTATION METHODS ===
    @api("bluetooth")
    def carcontrol_bluetooth_set_connection(self, connected: bool) -> Dict[str, Any]:
        """Set Bluetooth connection state."""
        self.is_connected = connected
        return {
            "success": True,
            "message": f"Bluetooth {'connected' if connected else 'disconnected'} successfully",
            "current_state": self.to_dict(),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert the Bluetooth state to a dictionary representation."""
        return {
            "is_connected": {
                "value": self.is_connected,
                "description": "Whether Bluetooth is connected",
                "type": type(self.is_connected).__name__,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Bluetooth":
        """Create a Bluetooth instance from a dictionary representation."""
        instance = cls()
        instance.is_connected = data["is_connected"]["value"]
        return instance
