from ..utils import api
from enum import Enum
from typing import Dict, Any


class Video:
    """
    Entity class representing the video system.
    """

    class VideoQuality(Enum):
        Q270P = "270p"
        Q480P = "480p"
        Q720P = "720p"
        Q1080P = "1080p"

    class Scene(Enum):
        FOREGROUND = "foreground"
        BACKGROUND = "background"

    PARAMS_DESCRIPTION = {
        "switch": {
            "description": "Power state of the video system.",
            "type": "bool",
            "required": True,
            "valid_values": [True, False],
            "value_meanings": {
                True: "Turn on the video system",
                False: "Turn off the video system",
            },
            "default": False,
            "notes": "All other video operations will automatically turn on the system if it was off.",
        },
        "video_name": {
            "description": "Name/title of the video to play. This is the primary video playback function.",
            "type": "str",
            "required": True,
            "default": "",
            "notes": "Simply provide the video name to start playing. System will auto turn on if needed.",
        },
        "quality": {
            "description": "Video playback quality/resolution.",
            "type": "str",
            "required": True,
            "valid_values": ["270p", "480p", "720p", "1080p"],
            "value_meanings": {
                "270p": "Low quality (270p) - saves data, suitable for slow networks",
                "480p": "Standard quality (480p) - balanced quality and data usage (default)",
                "720p": "HD quality (720p) - good quality for larger screens",
                "1080p": "Full HD quality (1080p) - best quality, uses more data",
            },
            "aliases": {
                "流畅 270p": "270p",
                "流畅 270P": "270p",
                "标清 480p": "480p",
                "标清 480P": "480p",
                "高清 720p": "720p",
                "高清 720P": "720p",
                "蓝光 1080p": "1080p",
                "蓝光 1080P": "1080p",
            },
            "default": "480p",
            "notes": "System will auto turn on if needed. Chinese aliases are supported.",
        },
        "is_fullscreen": {
            "description": "Whether to display video in fullscreen mode.",
            "type": "bool",
            "required": True,
            "valid_values": [True, False],
            "value_meanings": {
                True: "Enable fullscreen mode",
                False: "Disable fullscreen mode (windowed)",
            },
            "default": False,
            "notes": "System will auto turn on if needed.",
        },
        "scene": {
            "description": "Video playback scene/layer positioning.",
            "type": "str",
            "required": True,
            "valid_values": ["foreground", "background"],
            "value_meanings": {
                "foreground": "Play video in foreground - main display focus",
                "background": "Play video in background - allows other UI on top",
            },
            "aliases": {
                "前景": "foreground",
                "后景": "background",
            },
            "default": "foreground",
            "notes": "System will auto turn on if needed. Chinese aliases are supported.",
        },
        "volume": {
            "description": "Audio volume level for video playback.",
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
    }

    def __init__(self):
        """Initialize with default values."""
        self._is_on = False
        self._quality = self.VideoQuality.Q480P
        self._is_fullscreen = False
        self._scene = self.Scene.FOREGROUND
        self._volume = 50
        self._current_video = self._default_video()

    def _default_video(self) -> Dict[str, Any]:
        return {
            "name": "",
            "is_playing": False,
        }

    def _ensure_on(self) -> None:
        """Auto turn on video system if not already on."""
        if not self._is_on:
            self._is_on = True

    # === POWER STATE ===
    @property
    def is_on(self) -> bool:
        """Get video system state."""
        return self._is_on

    @is_on.setter
    def is_on(self, value: bool):
        """Set video system state."""
        self._is_on = bool(value)

    # === QUALITY ===
    @property
    def quality(self) -> "Video.VideoQuality":
        """Get current video quality."""
        return self._quality

    @quality.setter
    def quality(self, value: "Video.VideoQuality"):
        """Set current video quality."""
        self._quality = self._validate_quality(value)

    # === FULLSCREEN ===
    @property
    def is_fullscreen(self) -> bool:
        """Get fullscreen state."""
        return self._is_fullscreen

    @is_fullscreen.setter
    def is_fullscreen(self, value: bool):
        """Set fullscreen state."""
        self._is_fullscreen = bool(value)

    # === SCENE ===
    @property
    def scene(self) -> "Video.Scene":
        """Get video scene."""
        return self._scene

    @scene.setter
    def scene(self, value: "Video.Scene"):
        """Set video scene."""
        self._scene = self._validate_scene(value)

    # === VOLUME ===
    @property
    def volume(self) -> int:
        """Get volume (0-100)."""
        return self._volume

    @volume.setter
    def volume(self, value: int):
        """Set volume (0-100)."""
        self._volume = self._validate_range(value, 0, 100, "volume")

    # === CURRENT VIDEO ===
    @property
    def current_video(self) -> Dict[str, Any]:
        """Get current video info."""
        return dict(self._current_video)

    def _validate_quality(self, value: "Video.VideoQuality") -> "Video.VideoQuality":
        if isinstance(value, self.VideoQuality):
            return value
        if isinstance(value, str):
            mapping = {
                "270p": self.VideoQuality.Q270P,
                "480p": self.VideoQuality.Q480P,
                "720p": self.VideoQuality.Q720P,
                "1080p": self.VideoQuality.Q1080P,
                "流畅 270p": self.VideoQuality.Q270P,
                "流畅 270P": self.VideoQuality.Q270P,
                "标清 480p": self.VideoQuality.Q480P,
                "标清 480P": self.VideoQuality.Q480P,
                "高清 720p": self.VideoQuality.Q720P,
                "高清 720P": self.VideoQuality.Q720P,
                "蓝光 1080p": self.VideoQuality.Q1080P,
                "蓝光 1080P": self.VideoQuality.Q1080P,
            }
            normalized = value.strip()
            if normalized in mapping:
                return mapping[normalized]
        raise ValueError("quality must be 270p, 480p, 720p, or 1080p")

    def _validate_scene(self, value: "Video.Scene") -> "Video.Scene":
        if isinstance(value, self.Scene):
            return value
        if isinstance(value, str):
            mapping = {
                "foreground": self.Scene.FOREGROUND,
                "background": self.Scene.BACKGROUND,
                "前景": self.Scene.FOREGROUND,
                "后景": self.Scene.BACKGROUND,
            }
            if value in mapping:
                return mapping[value]
        raise ValueError("scene must be foreground or background")

    def _validate_range(self, value: int, min_value: int, max_value: int, name: str) -> int:
        if not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
        if value < min_value or value > max_value:
            raise ValueError(f"{name} must be between {min_value} and {max_value}")
        return value

    # === API IMPLEMENTATION METHODS ===
    @api("video")
    def carcontrol_video_switch(self, switch: bool) -> Dict[str, Any]:
        """Turn the video system on or off."""
        self.is_on = switch
        if not switch:
            self._current_video = self._default_video()
        return {
            "success": True,
            "message": f"Video system {'activated' if switch else 'deactivated'} successfully",
            "current_state": self.to_dict(),
        }

    @api("video")
    def carcontrol_video_play_video(self, video_name: str) -> Dict[str, Any]:
        """Play a video by name. This is the primary video playback function."""
        if not video_name or not video_name.strip():
            return {"success": False, "message": "video_name is required", "current_state": self.to_dict()}
        self._ensure_on()
        self._current_video = {
            "name": video_name.strip(),
            "is_playing": True,
        }
        return {
            "success": True,
            "message": f"Now playing: {video_name}",
            "current_state": self.to_dict(),
        }

    @api("video")
    def carcontrol_video_set_quality(self, quality: str) -> Dict[str, Any]:
        """Set video quality (270p/480p/720p/1080p)."""
        self._ensure_on()
        try:
            self.quality = quality
        except ValueError as exc:
            return {"success": False, "message": str(exc), "current_state": self.to_dict()}
        return {"success": True, "message": "Video quality set successfully", "current_state": self.to_dict()}

    @api("video")
    def carcontrol_video_set_fullscreen(self, fullscreen: bool) -> Dict[str, Any]:
        """Set fullscreen state."""
        self._ensure_on()
        self.is_fullscreen = fullscreen
        return {"success": True, "message": "Video fullscreen set successfully", "current_state": self.to_dict()}

    @api("video")
    def carcontrol_video_set_scene(self, scene: str) -> Dict[str, Any]:
        """Set video scene (foreground/background)."""
        self._ensure_on()
        try:
            self.scene = scene
        except ValueError as exc:
            return {"success": False, "message": str(exc), "current_state": self.to_dict()}
        return {"success": True, "message": "Video scene set successfully", "current_state": self.to_dict()}

    @api("video")
    def carcontrol_video_set_volume(self, volume: int) -> Dict[str, Any]:
        """Set volume (0-100)."""
        self._ensure_on()
        try:
            self.volume = volume
        except ValueError as exc:
            return {"success": False, "message": str(exc), "current_state": self.to_dict()}
        return {"success": True, "message": "Video volume set successfully", "current_state": self.to_dict()}

    def to_dict(self) -> Dict[str, Any]:
        """Convert the video system to a dictionary representation."""
        return {
            "is_on": {
                "value": self.is_on,
                "description": "Whether the video system is on",
                "type": type(self.is_on).__name__,
            },
            "quality": {
                "value": self.quality.value,
                "description": "Current video quality",
                "type": type(self.quality.value).__name__,
            },
            "is_fullscreen": {
                "value": self.is_fullscreen,
                "description": "Whether the video is in fullscreen",
                "type": type(self.is_fullscreen).__name__,
            },
            "scene": {
                "value": self.scene.value,
                "description": "Current video scene",
                "type": type(self.scene.value).__name__,
            },
            "volume": {
                "value": self.volume,
                "description": "Video volume (0-100)",
                "type": type(self.volume).__name__,
            },
            "current_video": {
                "value": dict(self._current_video),
                "description": "Current video info (name, is_playing)",
                "type": type(self._current_video).__name__,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Video":
        """Create an instance from a dictionary representation."""
        instance = cls()
        instance.is_on = data["is_on"]["value"]
        instance.quality = data["quality"]["value"]
        instance.is_fullscreen = data["is_fullscreen"]["value"]
        instance.scene = data["scene"]["value"]
        instance.volume = data["volume"]["value"]
        instance._current_video = dict(data["current_video"]["value"])
        return instance
