from ..utils import api
from typing import Dict, Any, Optional


class Radio:
    """
    Entity class representing the radio system.
    """

    PARAMS_DESCRIPTION = {
        "switch": {
            "description": "Power state of the radio system.",
            "type": "bool",
            "required": True,
            "valid_values": [True, False],
            "value_meanings": {
                True: "Turn on the radio",
                False: "Turn off the radio",
            },
            "default": False,
            "notes": "All other radio operations will automatically turn on the radio if it was off.",
        },
        "volume": {
            "description": "Audio volume level for radio playback.",
            "type": "int",
            "required": True,
            "valid_range": {"min": 0, "max": 100},
            "value_meanings": {
                0: "Muted",
                50: "Medium volume (default)",
                100: "Maximum volume",
            },
            "default": 50,
            "notes": "System will auto turn on if needed.",
        },
        "station_name": {
            "description": "Name of the radio station to tune to.",
            "type": "str",
            "required": False,
            "default": "",
            "notes": "Either station_name or frequency should be provided. System will auto turn on if needed.",
        },
        "frequency": {
            "description": "Frequency of the radio station to tune to (e.g., '98.7MHz').",
            "type": "str",
            "required": False,
            "default": "",
            "notes": "Either station_name or frequency should be provided. System will auto turn on if needed.",
        },
    }

    def __init__(self):
        """Initialize with default values."""
        self._is_on = False
        self._volume = 50
        self._current_station: Dict[str, Any] = {}
        self._stations = self._default_stations()

    def _default_stations(self):
        return [
            {"name": "News FM", "frequency": "98.7MHz"},
            {"name": "Music FM", "frequency": "103.2MHz"},
        ]

    # === POWER STATE ===
    @property
    def is_on(self) -> bool:
        """Get radio power state."""
        return self._is_on

    @is_on.setter
    def is_on(self, value: bool):
        """Set radio power state."""
        self._is_on = bool(value)

    # === VOLUME ===
    @property
    def volume(self) -> int:
        """Get volume (0-100)."""
        return self._volume

    @volume.setter
    def volume(self, value: int):
        """Set volume (0-100)."""
        self._volume = self._validate_range(value, 0, 100, "volume")

    # === CURRENT STATION ===
    @property
    def current_station(self) -> Dict[str, Any]:
        """Get current station info."""
        return dict(self._current_station)

    def _validate_range(self, value: int, min_value: int, max_value: int, name: str) -> int:
        if not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
        if value < min_value or value > max_value:
            raise ValueError(f"{name} must be between {min_value} and {max_value}")
        return value

    def _find_station(self, name: Optional[str], frequency: Optional[str]) -> Dict[str, Any]:
        for station in self._stations:
            if name and station["name"] == name:
                return dict(station)
            if frequency and station["frequency"] == frequency:
                return dict(station)
        if name or frequency:
            return {
                "name": name or "",
                "frequency": frequency or "",
            }
        raise ValueError("station name or frequency must be provided")

    def _ensure_on(self) -> None:
        """Auto turn on radio if not already on."""
        if not self._is_on:
            self._is_on = True

    # === API IMPLEMENTATION METHODS ===
    @api("radio")
    def carcontrol_radio_switch(self, switch: bool) -> Dict[str, Any]:
        """Turn the radio on or off."""
        self.is_on = switch
        return {
            "success": True,
            "message": f"Radio {'activated' if switch else 'deactivated'} successfully",
            "current_state": self.to_dict(),
        }

    @api("radio")
    def carcontrol_radio_set_volume(self, volume: int) -> Dict[str, Any]:
        """Set radio volume (0-100)."""
        self._ensure_on()
        try:
            self.volume = volume
        except ValueError as exc:
            return {"success": False, "message": str(exc), "current_state": self.to_dict()}
        return {"success": True, "message": "Radio volume set successfully", "current_state": self.to_dict()}

    @api("radio")
    def carcontrol_radio_play_station(self, name: Optional[str] = None, frequency: Optional[str] = None) -> Dict[str, Any]:
        """Play a station by name or frequency."""
        self._ensure_on()
        try:
            self._current_station = self._find_station(name, frequency)
        except ValueError as exc:
            return {"success": False, "message": str(exc), "current_state": self.to_dict()}
        return {
            "success": True,
            "message": "Radio station set successfully",
            "current_state": self.to_dict(),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert the radio to a dictionary representation."""
        return {
            "is_on": {
                "value": self.is_on,
                "description": "Whether the radio is on",
                "type": type(self.is_on).__name__,
            },
            "volume": {
                "value": self.volume,
                "description": "Radio volume (0-100)",
                "type": type(self.volume).__name__,
            },
            "current_station": {
                "value": dict(self._current_station),
                "description": "Current radio station info",
                "type": type(self._current_station).__name__,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Radio":
        """Create an instance from a dictionary representation."""
        instance = cls()
        instance.is_on = data["is_on"]["value"]
        instance.volume = data["volume"]["value"]
        instance._current_station = dict(data["current_station"]["value"])
        return instance
