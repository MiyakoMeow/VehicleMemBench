from ..utils import api
from enum import Enum
from typing import Dict, Any, Optional


class Navigation:
    """
    Entity class representing the navigation system.
    """

    class VoiceMode(Enum):
        SIMPLE = "simple"
        DETAILED = "detailed"
        MUTE = "mute"

    class MapView(Enum):
        VIEW_2D = "2d"
        VIEW_3D = "3d"
        HEADING_UP = "heading_up"
        NORTH_UP = "north_up"

    class RoutePreference(Enum):
        FASTEST = "fastest"
        SHORTEST = "shortest"
        NO_HIGHWAY = "no_highway"
        NO_TOLL = "no_toll"

    PARAMS_DESCRIPTION = {
        "switch": {
            "description": "Power state of the navigation system.",
            "type": "bool",
            "required": True,
            "valid_values": [True, False],
            "value_meanings": {
                True: "Turn on the navigation system",
                False: "Turn off the navigation system",
            },
            "default": False,
            "notes": "All other navigation operations will automatically turn on the system if it was off.",
        },
        "destination": {
            "description": "The destination address or location name to navigate to. This is the primary navigation function.",
            "type": "str",
            "required": True,
            "default": "",
            "notes": "Simply provide the destination name to start navigation. The system will automatically turn on and calculate the route. This is the most important function of the navigation system.",
        },
        "route_preference": {
            "description": "Route calculation preference that affects how the route is planned.",
            "type": "str",
            "required": False,
            "valid_values": ["fastest", "shortest", "no_highway", "no_toll"],
            "value_meanings": {
                "fastest": "Time priority - find the fastest route (default)",
                "shortest": "Distance priority - find the shortest route",
                "no_highway": "Avoid highways - use regular roads only",
                "no_toll": "Avoid tolls - minimize toll road usage",
            },
            "default": "fastest",
            "notes": "Optional parameter when setting destination. System will auto turn on if needed.",
        },
        "voice_mode": {
            "description": "Voice guidance mode for turn-by-turn navigation instructions.",
            "type": "str",
            "required": True,
            "valid_values": ["simple", "detailed", "mute"],
            "value_meanings": {
                "simple": "Brief voice prompts - basic turn instructions only",
                "detailed": "Full voice guidance - includes road names, distances, lane guidance (default)",
                "mute": "Silent mode - no voice guidance, visual only",
            },
            "aliases": {
                "简洁": "simple",
                "详细": "detailed",
                "静音": "mute",
            },
            "default": "detailed",
            "notes": "Chinese aliases are supported. System will auto turn on if needed.",
        },
        "volume": {
            "description": "Voice guidance volume level.",
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
        "map_view": {
            "description": "Map display perspective/orientation.",
            "type": "str",
            "required": True,
            "valid_values": ["2d", "3d", "heading_up", "north_up"],
            "value_meanings": {
                "2d": "2D flat map view (default)",
                "3d": "3D perspective map view",
                "heading_up": "Map rotates so vehicle direction is always up",
                "north_up": "Map fixed with north at top",
            },
            "default": "2d",
            "notes": "System will auto turn on if needed.",
        },
        "map_zoom": {
            "description": "Map zoom level controlling the scale of the displayed area.",
            "type": "int",
            "required": True,
            "valid_range": {"min": 1, "max": 10},
            "value_meanings": {
                1: "Most zoomed out - large area view",
                5: "Medium zoom (default)",
                10: "Most zoomed in - detailed street view",
            },
            "default": 5,
            "notes": "System will auto turn on if needed.",
        },
        "traffic_display": {
            "description": "Whether to show real-time traffic conditions on the map.",
            "type": "bool",
            "required": True,
            "valid_values": [True, False],
            "value_meanings": {
                True: "Show traffic layer - displays congestion with colors (default)",
                False: "Hide traffic layer",
            },
            "default": True,
            "notes": "System will auto turn on if needed.",
        },
        "speed_camera_alert": {
            "description": "Whether to alert about speed cameras and speed traps.",
            "type": "bool",
            "required": True,
            "valid_values": [True, False],
            "value_meanings": {
                True: "Enable alerts - warn about upcoming speed cameras (default)",
                False: "Disable alerts",
            },
            "default": True,
            "notes": "System will auto turn on if needed.",
        },
    }

    def __init__(self):
        """Initialize with default values."""
        self._is_on = False
        self._voice_mode = self.VoiceMode.DETAILED
        self._volume = 50
        self._map_view = self.MapView.VIEW_2D
        self._map_zoom = 5
        self._traffic_display_enabled = True
        self._speed_camera_enabled = True
        self._current_destination = self._default_destination()

    def _default_destination(self) -> Dict[str, Any]:
        return {
            "name": "",
            "route_preference": "fastest",
            "is_navigating": False,
        }

    def _ensure_on(self) -> None:
        """Auto turn on navigation system if not already on."""
        if not self._is_on:
            self._is_on = True

    # === DEVICE STATE ===
    @property
    def is_on(self) -> bool:
        """Get navigation device state."""
        return self._is_on

    @is_on.setter
    def is_on(self, value: bool):
        """Set navigation device state."""
        self._is_on = bool(value)

    # === VOICE MODE ===
    @property
    def voice_mode(self) -> "Navigation.VoiceMode":
        """Get voice broadcast mode."""
        return self._voice_mode

    @voice_mode.setter
    def voice_mode(self, value: "Navigation.VoiceMode"):
        """Set voice broadcast mode."""
        self._voice_mode = self._validate_voice_mode(value)

    # === VOLUME ===
    @property
    def volume(self) -> int:
        """Get volume (0-100)."""
        return self._volume

    @volume.setter
    def volume(self, value: int):
        """Set volume (0-100)."""
        self._volume = self._validate_range(value, 0, 100, "volume")

    # === MAP VIEW ===
    @property
    def map_view(self) -> "Navigation.MapView":
        """Get map view."""
        return self._map_view

    @map_view.setter
    def map_view(self, value: "Navigation.MapView"):
        """Set map view."""
        self._map_view = self._validate_map_view(value)

    # === MAP ZOOM ===
    @property
    def map_zoom(self) -> int:
        """Get map zoom (1-10)."""
        return self._map_zoom

    @map_zoom.setter
    def map_zoom(self, value: int):
        """Set map zoom (1-10)."""
        self._map_zoom = self._validate_range(value, 1, 10, "map_zoom")

    # === TRAFFIC DISPLAY ===
    @property
    def traffic_display_enabled(self) -> bool:
        """Get traffic display switch."""
        return self._traffic_display_enabled

    @traffic_display_enabled.setter
    def traffic_display_enabled(self, value: bool):
        """Set traffic display switch."""
        self._traffic_display_enabled = bool(value)

    # === SPEED CAMERA ===
    @property
    def speed_camera_enabled(self) -> bool:
        """Get speed camera switch."""
        return self._speed_camera_enabled

    @speed_camera_enabled.setter
    def speed_camera_enabled(self, value: bool):
        """Set speed camera switch."""
        self._speed_camera_enabled = bool(value)

    # === CURRENT DESTINATION ===
    @property
    def current_destination(self) -> Dict[str, Any]:
        """Get current destination info."""
        return dict(self._current_destination)

    def _validate_voice_mode(self, value) -> "Navigation.VoiceMode":
        if isinstance(value, self.VoiceMode):
            return value
        if isinstance(value, str):
            mapping = {
                "simple": self.VoiceMode.SIMPLE,
                "detailed": self.VoiceMode.DETAILED,
                "mute": self.VoiceMode.MUTE,
                "简洁": self.VoiceMode.SIMPLE,
                "详细": self.VoiceMode.DETAILED,
                "静音": self.VoiceMode.MUTE,
            }
            if value in mapping:
                return mapping[value]
        raise ValueError("voice_mode must be simple, detailed, or mute")

    def _validate_map_view(self, value) -> "Navigation.MapView":
        if isinstance(value, self.MapView):
            return value
        if isinstance(value, str):
            for item in self.MapView:
                if item.value == value:
                    return item
        raise ValueError("map_view must be 2d, 3d, heading_up, or north_up")

    def _validate_route_preference(self, value) -> "Navigation.RoutePreference":
        if isinstance(value, self.RoutePreference):
            return value
        if isinstance(value, str):
            mapping = {
                "fastest": self.RoutePreference.FASTEST,
                "shortest": self.RoutePreference.SHORTEST,
                "no_highway": self.RoutePreference.NO_HIGHWAY,
                "no_toll": self.RoutePreference.NO_TOLL,
                "最快": self.RoutePreference.FASTEST,
                "最短": self.RoutePreference.SHORTEST,
                "不走高速": self.RoutePreference.NO_HIGHWAY,
                "避免收费": self.RoutePreference.NO_TOLL,
            }
            if value in mapping:
                return mapping[value]
        raise ValueError("route_preference must be fastest, shortest, no_highway, or no_toll")

    def _validate_range(self, value: int, min_value: int, max_value: int, name: str) -> int:
        if not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
        if value < min_value or value > max_value:
            raise ValueError(f"{name} must be between {min_value} and {max_value}")
        return value

    # === API IMPLEMENTATION METHODS ===
    @api("navigation")
    def carcontrol_navigation_switch(self, switch: bool) -> Dict[str, Any]:
        """Turn the navigation device on or off."""
        self.is_on = switch
        if not switch:
            # Clear destination when turning off
            self._current_destination = self._default_destination()
        return {
            "success": True,
            "message": f"Navigation {'activated' if switch else 'deactivated'} successfully",
            "current_state": self.to_dict(),
        }

    @api("navigation")
    def carcontrol_navigation_navigate_to(self, destination: str, route_preference: Optional[str] = None) -> Dict[str, Any]:
        """Navigate to a destination. This is the primary navigation function."""
        if not destination or not destination.strip():
            return {"success": False, "message": "destination is required", "current_state": self.to_dict()}

        self._ensure_on()

        # Validate route preference if provided
        pref = "fastest"
        if route_preference:
            try:
                pref = self._validate_route_preference(route_preference).value
            except ValueError as exc:
                return {"success": False, "message": str(exc), "current_state": self.to_dict()}

        self._current_destination = {
            "name": destination.strip(),
            "route_preference": pref,
            "is_navigating": True,
        }

        return {
            "success": True,
            "message": f"Navigating to: {destination}" + (f" (preference: {pref})" if route_preference else ""),
            "current_state": self.to_dict(),
        }

    @api("navigation")
    def carcontrol_navigation_stop(self) -> Dict[str, Any]:
        """Stop current navigation and clear destination."""
        self._current_destination = self._default_destination()
        return {
            "success": True,
            "message": "Navigation stopped",
            "current_state": self.to_dict(),
        }

    @api("navigation")
    def carcontrol_navigation_set_voice_mode(self, mode: str) -> Dict[str, Any]:
        """Set voice guidance mode (simple/detailed/mute)."""
        self._ensure_on()
        try:
            self.voice_mode = mode
        except ValueError as exc:
            return {"success": False, "message": str(exc), "current_state": self.to_dict()}
        return {"success": True, "message": f"Voice mode set to {self.voice_mode.value}", "current_state": self.to_dict()}

    @api("navigation")
    def carcontrol_navigation_set_volume(self, volume: int) -> Dict[str, Any]:
        """Set voice volume (0-100)."""
        self._ensure_on()
        try:
            self.volume = volume
        except ValueError as exc:
            return {"success": False, "message": str(exc), "current_state": self.to_dict()}
        return {"success": True, "message": f"Volume set to {volume}", "current_state": self.to_dict()}

    @api("navigation")
    def carcontrol_navigation_set_map_view(self, view: str) -> Dict[str, Any]:
        """Set map view (2d/3d/heading_up/north_up)."""
        self._ensure_on()
        try:
            self.map_view = view
        except ValueError as exc:
            return {"success": False, "message": str(exc), "current_state": self.to_dict()}
        return {"success": True, "message": f"Map view set to {self.map_view.value}", "current_state": self.to_dict()}

    @api("navigation")
    def carcontrol_navigation_set_map_zoom(self, zoom: int) -> Dict[str, Any]:
        """Set map zoom (1-10)."""
        self._ensure_on()
        try:
            self.map_zoom = zoom
        except ValueError as exc:
            return {"success": False, "message": str(exc), "current_state": self.to_dict()}
        return {"success": True, "message": f"Map zoom set to {zoom}", "current_state": self.to_dict()}

    @api("navigation")
    def carcontrol_navigation_set_traffic_display(self, enabled: bool) -> Dict[str, Any]:
        """Enable or disable traffic display on map."""
        self._ensure_on()
        self.traffic_display_enabled = enabled
        return {
            "success": True,
            "message": f"Traffic display {'enabled' if enabled else 'disabled'}",
            "current_state": self.to_dict(),
        }

    @api("navigation")
    def carcontrol_navigation_set_speed_camera_alert(self, enabled: bool) -> Dict[str, Any]:
        """Enable or disable speed camera alerts."""
        self._ensure_on()
        self.speed_camera_enabled = enabled
        return {
            "success": True,
            "message": f"Speed camera alert {'enabled' if enabled else 'disabled'}",
            "current_state": self.to_dict(),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert the navigation system to a dictionary representation."""
        return {
            "is_on": {
                "value": self.is_on,
                "description": "Whether the navigation device is on",
                "type": type(self.is_on).__name__,
            },
            "voice_mode": {
                "value": self.voice_mode.value,
                "description": "Voice guidance mode",
                "type": type(self.voice_mode.value).__name__,
            },
            "volume": {
                "value": self.volume,
                "description": "Voice volume (0-100)",
                "type": type(self.volume).__name__,
            },
            "map_view": {
                "value": self.map_view.value,
                "description": "Map view mode",
                "type": type(self.map_view.value).__name__,
            },
            "map_zoom": {
                "value": self.map_zoom,
                "description": "Map zoom (1-10)",
                "type": type(self.map_zoom).__name__,
            },
            "traffic_display_enabled": {
                "value": self.traffic_display_enabled,
                "description": "Whether traffic display is enabled",
                "type": type(self.traffic_display_enabled).__name__,
            },
            "speed_camera_enabled": {
                "value": self.speed_camera_enabled,
                "description": "Whether speed camera alerts are enabled",
                "type": type(self.speed_camera_enabled).__name__,
            },
            "current_destination": {
                "value": dict(self._current_destination),
                "description": "Current destination info (name, route_preference, is_navigating)",
                "type": type(self._current_destination).__name__,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Navigation":
        """Create an instance from a dictionary representation."""
        instance = cls()
        instance.is_on = data["is_on"]["value"]
        instance.voice_mode = data["voice_mode"]["value"]
        instance.volume = data["volume"]["value"]
        instance.map_view = data["map_view"]["value"]
        instance.map_zoom = data["map_zoom"]["value"]
        instance.traffic_display_enabled = data["traffic_display_enabled"]["value"]
        instance.speed_camera_enabled = data["speed_camera_enabled"]["value"]
        instance._current_destination = dict(data["current_destination"]["value"])
        return instance
