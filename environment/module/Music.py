from ..utils import api
from enum import Enum
from typing import Dict, Any, List, Optional


class Music:
    """
    Entity class representing the music system.
    """

    class PlayMode(Enum):
        SINGLE_LOOP = "single_loop"
        SHUFFLE = "shuffle"
        SEQUENTIAL = "sequential"
        LIST_LOOP = "list_loop"

    PARAMS_DESCRIPTION = {
        "switch": {
            "description": "Power state of the music system.",
            "type": "bool",
            "required": True,
            "valid_values": [True, False],
            "value_meanings": {
                True: "Turn on the music system",
                False: "Turn off the music system",
            },
            "default": False,
            "notes": "All other music operations will automatically turn on the system if it was off.",
        },
        "song_name": {
            "description": "Name of the song to play. This is the primary way to request music playback.",
            "type": "str",
            "required": True,
            "default": "",
            "notes": "Simply provide the song name to start playing. The system will automatically turn on if needed. This is the most important function of the music system.",
        },
        "artist": {
            "description": "Optional artist name to help identify the correct song when multiple songs share the same name.",
            "type": "str",
            "required": False,
            "default": "",
            "notes": "Use this parameter when you want to play a specific version of a song by a particular artist.",
        },
        "play_mode": {
            "description": "Playback mode that controls how songs are played in sequence.",
            "type": "str",
            "required": True,
            "valid_values": ["single_loop", "shuffle", "sequential", "list_loop"],
            "value_meanings": {
                "single_loop": "Repeat the current song continuously",
                "shuffle": "Play songs in random order",
                "sequential": "Play songs in list order (default)",
                "list_loop": "Loop through the entire playlist",
            },
            "aliases": {
                "单曲循环": "single_loop",
                "随机播放": "shuffle",
                "顺序播放": "sequential",
                "列表循环": "list_loop",
            },
            "default": "sequential",
            "notes": "Chinese aliases are supported. System will auto turn on if needed.",
        },
        "volume": {
            "description": "Audio volume level for music playback.",
            "type": "int",
            "required": True,
            "valid_range": {"min": 0, "max": 100},
            "value_meanings": {
                0: "Muted",
                50: "Medium volume (default)",
                100: "Maximum volume",
            },
            "default": 50,
            "notes": "System will auto turn on if needed when setting volume.",
        },
        "lyrics_enabled": {
            "description": "Whether to display song lyrics on screen during playback.",
            "type": "bool",
            "required": True,
            "valid_values": [True, False],
            "value_meanings": {
                True: "Show lyrics on display",
                False: "Hide lyrics",
            },
            "default": False,
            "notes": "System will auto turn on if needed.",
        },
        "favorite": {
            "description": "Whether to add or remove the current song from favorites.",
            "type": "bool",
            "required": True,
            "valid_values": [True, False],
            "value_meanings": {
                True: "Add current song to favorites",
                False: "Remove current song from favorites",
            },
            "default": False,
            "notes": "System will auto turn on if needed. Only works when a song is currently playing.",
        },
    }

    def __init__(self):
        """Initialize with default values."""
        self._is_on = False
        self._play_mode = self.PlayMode.SEQUENTIAL
        self._lyrics_display_enabled = False
        self._volume = 50
        self._current_song = self._default_song()
        self._favorite_list = []

    def _default_song(self) -> Dict[str, Any]:
        return {
            "name": "",
            "artist": "",
            "is_favorited": False,
        }

    def _ensure_on(self) -> None:
        """Auto turn on music system if not already on."""
        if not self._is_on:
            self._is_on = True

    # === POWER STATE ===
    @property
    def is_on(self) -> bool:
        """Get music system state."""
        return self._is_on

    @is_on.setter
    def is_on(self, value: bool):
        """Set music system state."""
        self._is_on = bool(value)

    # === PLAY MODE ===
    @property
    def play_mode(self) -> "Music.PlayMode":
        """Get play mode."""
        return self._play_mode

    @play_mode.setter
    def play_mode(self, value: "Music.PlayMode"):
        """Set play mode."""
        self._play_mode = self._validate_play_mode(value)

    # === LYRICS DISPLAY ===
    @property
    def lyrics_display_enabled(self) -> bool:
        """Get lyrics display state."""
        return self._lyrics_display_enabled

    @lyrics_display_enabled.setter
    def lyrics_display_enabled(self, value: bool):
        """Set lyrics display state."""
        self._lyrics_display_enabled = bool(value)

    # === VOLUME ===
    @property
    def volume(self) -> int:
        """Get volume (0-100)."""
        return self._volume

    @volume.setter
    def volume(self, value: int):
        """Set volume (0-100)."""
        self._volume = self._validate_range(value, 0, 100, "volume")

    # === CURRENT SONG ===
    @property
    def current_song(self) -> Dict[str, Any]:
        """Get current song info."""
        return dict(self._current_song)

    # === FAVORITE LIST ===
    @property
    def favorite_list(self) -> List[str]:
        """Get favorite song list."""
        return list(self._favorite_list)

    def _validate_play_mode(self, value: "Music.PlayMode") -> "Music.PlayMode":
        if isinstance(value, self.PlayMode):
            return value
        if isinstance(value, str):
            mapping = {
                "single_loop": self.PlayMode.SINGLE_LOOP,
                "shuffle": self.PlayMode.SHUFFLE,
                "sequential": self.PlayMode.SEQUENTIAL,
                "list_loop": self.PlayMode.LIST_LOOP,
                "单曲循环": self.PlayMode.SINGLE_LOOP,
                "随机播放": self.PlayMode.SHUFFLE,
                "顺序播放": self.PlayMode.SEQUENTIAL,
                "列表循环": self.PlayMode.LIST_LOOP,
            }
            if value in mapping:
                return mapping[value]
        raise ValueError("play_mode must be single_loop, shuffle, sequential, or list_loop")

    def _validate_range(self, value: int, min_value: int, max_value: int, name: str) -> int:
        if not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
        if value < min_value or value > max_value:
            raise ValueError(f"{name} must be between {min_value} and {max_value}")
        return value

    # === API IMPLEMENTATION METHODS ===
    @api("music")
    def carcontrol_music_switch(self, switch: bool) -> Dict[str, Any]:
        """Turn the music system on or off."""
        self.is_on = switch
        return {
            "success": True,
            "message": f"Music system {'activated' if switch else 'deactivated'} successfully",
            "current_state": self.to_dict(),
        }

    @api("music")
    def carcontrol_music_play_song(self, song_name: str, artist: Optional[str] = None) -> Dict[str, Any]:
        """Play a song by name. This is the primary music playback function."""
        if not song_name or not song_name.strip():
            return {"success": False, "message": "song_name is required", "current_state": self.to_dict()}
        self._ensure_on()
        self._current_song = {
            "name": song_name.strip(),
            "artist": artist.strip() if artist else "",
            "is_favorited": song_name in self._favorite_list,
        }
        return {
            "success": True,
            "message": f"Now playing: {song_name}" + (f" by {artist}" if artist else ""),
            "current_state": self.to_dict(),
        }

    @api("music")
    def carcontrol_music_set_play_mode(self, mode: str) -> Dict[str, Any]:
        """Set play mode (single_loop/shuffle/sequential/list_loop)."""
        self._ensure_on()
        try:
            self.play_mode = mode
        except ValueError as exc:
            return {"success": False, "message": str(exc), "current_state": self.to_dict()}
        return {"success": True, "message": f"Play mode set to {self.play_mode.value}", "current_state": self.to_dict()}

    @api("music")
    def carcontrol_music_set_volume(self, volume: int) -> Dict[str, Any]:
        """Set volume (0-100)."""
        self._ensure_on()
        try:
            self.volume = volume
        except ValueError as exc:
            return {"success": False, "message": str(exc), "current_state": self.to_dict()}
        return {"success": True, "message": f"Volume set to {volume}", "current_state": self.to_dict()}

    @api("music")
    def carcontrol_music_set_lyrics_display(self, enabled: bool) -> Dict[str, Any]:
        """Enable or disable lyrics display."""
        self._ensure_on()
        self.lyrics_display_enabled = enabled
        return {
            "success": True,
            "message": f"Lyrics display {'enabled' if enabled else 'disabled'}",
            "current_state": self.to_dict(),
        }

    @api("music")
    def carcontrol_music_set_favorite(self, favorite: bool) -> Dict[str, Any]:
        """Add or remove current song from favorites."""
        self._ensure_on()
        if not self._current_song["name"]:
            return {"success": False, "message": "No song is currently playing", "current_state": self.to_dict()}

        song_name = self._current_song["name"]
        if favorite:
            if song_name not in self._favorite_list:
                self._favorite_list.append(song_name)
            self._current_song["is_favorited"] = True
            return {"success": True, "message": f"Added '{song_name}' to favorites", "current_state": self.to_dict()}
        else:
            if song_name in self._favorite_list:
                self._favorite_list.remove(song_name)
            self._current_song["is_favorited"] = False
            return {"success": True, "message": f"Removed '{song_name}' from favorites", "current_state": self.to_dict()}

    def to_dict(self) -> Dict[str, Any]:
        """Convert the music system to a dictionary representation."""
        return {
            "is_on": {
                "value": self.is_on,
                "description": "Whether the music system is on",
                "type": type(self.is_on).__name__,
            },
            "play_mode": {
                "value": self.play_mode.value,
                "description": "Music play mode",
                "type": type(self.play_mode.value).__name__,
            },
            "volume": {
                "value": self.volume,
                "description": "Music volume (0-100)",
                "type": type(self.volume).__name__,
            },
            "lyrics_display_enabled": {
                "value": self.lyrics_display_enabled,
                "description": "Whether lyrics display is enabled",
                "type": type(self.lyrics_display_enabled).__name__,
            },
            "current_song": {
                "value": dict(self._current_song),
                "description": "Current song info (name, artist, is_favorited)",
                "type": type(self._current_song).__name__,
            },
            "favorite_list": {
                "value": list(self._favorite_list),
                "description": "List of favorited song names",
                "type": type(self._favorite_list).__name__,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Music":
        """Create an instance from a dictionary representation."""
        instance = cls()
        instance.is_on = data["is_on"]["value"]
        instance.play_mode = data["play_mode"]["value"]
        instance.volume = data["volume"]["value"]
        instance.lyrics_display_enabled = data["lyrics_display_enabled"]["value"]
        instance._current_song = dict(data["current_song"]["value"])
        instance._favorite_list = list(data.get("favorite_list", {}).get("value", []))
        return instance
