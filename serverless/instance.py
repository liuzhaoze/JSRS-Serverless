from enum import Enum


class InstanceType(Enum):
    nano = 0
    micro = 1
    small = 2
    medium = 3
    large = 4
    xlarge = 5
    xxlarge = 6


class Zone(Enum):
    us_east_1 = 0
    us_east_2 = 1
    us_west_1 = 2
    us_west_2 = 3
