from .module import *
from .utils import modules_dict, capitalize_first


class VehicleWorld:
    def __init__(self):
        self.HUD = HUD() # done
        self.centerInformationDisplay = CenterInformationDisplay() # done
        self.instrumentPanel = InstrumentPanel() # done
        self.frontTrunk = FrontTrunk() # done
        self.trunk = Trunk() # done
        self.fuelPort = FuelPort() # done
        self.rearviewMirror = RearviewMirror() # done
        self.sunroof = Sunroof() # done
        self.navigation = Navigation() # done
        self.seat = Seat() # done
        self.radio = Radio() # done
        self.airConditioner = AirConditioner() # done
        self.footPedal = FootPedal() # done
        self.bluetooth = Bluetooth() # done
        self.video = Video() # done
        self.window = Window() # done
        self.door = Door() # done
        self.sunshade = Sunshade() # done
        self.wiper = Wiper() # done
        self.music = Music() # done
        self.overheadScreen = OverheadScreen() # done
        self.steeringWheel = SteeringWheel() # done

        # Lighting
        self.light = Light() # done
            
    def to_dict(self):
        data = {}

        for key, description in modules_dict.items():
            module = getattr(self, key, None)
            if module is None:
                continue
            data[key] = {
                "value": module.to_dict(),
                "description": description,
                "type": type(module).__name__
            }

        return data

    @classmethod
    def from_dict(cls, data):
        """
        Restore VehicleWorld object from dictionary form.
        Missing fields will be initialized using default constructor.
        """
        vehicle_world = cls()

        for key in modules_dict:
            module_data = data.get(key, {}).get("value")
            if module_data is None:
                continue
            class_name = capitalize_first(key)
            klass = globals().get(class_name)
            if not klass:
                raise ValueError(f"Class {class_name} not found")
            setattr(vehicle_world, key, klass.from_dict(module_data))

        return vehicle_world
