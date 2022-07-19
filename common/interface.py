import dataclasses as dc
import enum
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import numpy.typing as npt


class Error(enum.IntEnum):
    NO_ERROR = 0
    NEED_RESET = 1
    X_OVERFLOW = 2
    V_OVERFLOW = 3
    A_OVERFLOW = 4
    MOTOR_STALLED = 5
    ENDSTOP_HIT = 6

    def __bool__(self) -> bool:
        return self != Error.NO_ERROR


@dc.dataclass
class Config:
    # software cart limits
    max_position: float = 0.25  # m
    max_velocity: float = 25.0  # m/s
    max_acceleration: float = 7  # m/s^2
    # hardware limits
    hard_max_position: float = 4  # m
    hard_max_velocity: float = 25  # m/s
    hard_max_acceleration: float = 8  # m/s^2
    # hardware flags
    clamp_position: bool = False
    clamp_velocity: bool = False
    clamp_acceleration: bool = False
    # physical params
    pole_length: float = 0.2  # m
    pole_mass: float = 0.118  # kg
    gravity: float = 9.8  # m/s^2
    friction_coefficient: float = 0


@dc.dataclass
class State:
    cart_position: float = 0
    cart_velocity: float = 0
    pole_angle: float = 0
    pole_angular_velocity: float = 0
    error: Error = Error.NO_ERROR
    cart_acceleration: float = 0

    @staticmethod
    def from_array(a: npt.NDArray[float]) -> "State":
        """
        q = (x, a, v, w)
        """
        return State(a[0], a[2], a[1], a[3])

    @staticmethod
    def home() -> "State":
        return State(0.0, 0.0, 0.0, 0.0)

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return (
            self.cart_position,
            self.pole_angle,
            self.cart_velocity,
            self.pole_angular_velocity,
        )

    def as_array(self) -> npt.NDArray[float]:
        return np.array(self.as_tuple())

    def as_array_4x1(self) -> npt.NDArray[float]:
        return self.as_array().reshape(4, 1)

    def __repr__(self) -> str:
        return "(x={x:+.2f}, v={v:+.2f}, a={a:+.2f}, w={w:+.2f}, err={err})".format(
            x=self.cart_position,
            v=self.cart_velocity,
            a=self.pole_angle,
            w=self.pole_angular_velocity,
            err=self.error,
        )


class CartPoleBase(ABC):
    """
    Description:
        Сlass implements a physical simulation of the cart-pole device.
        A pole is attached by an joint to a cart, which moves along guide axis.
        The pendulum is initially at rest state. The goal is to maintain it in
        upright pose by increasing and reducing the cart's velocity.
    Source:
        This environment is some variation of the cart-pole problem
        described by Barto, Sutton, and Anderson
    Initial state:
        A pole is at starting position 0 with no velocity and acceleration.
    """

    @abstractmethod
    def reset(self, config: Config) -> None:
        """
        Resets the device to the initial state.
        The pole is at rest position and cart is centered.
        It must be called at the beginning of any session.
        """
        pass

    @abstractmethod
    def get_state(self) -> State:
        """
        Returns current device state.
        """
        pass

    @abstractmethod
    def get_info(self) -> dict:
        """
        Returns usefull debug information.
        """
        pass

    @abstractmethod
    def get_target(self) -> float:
        """
        Returns current target acceleration.
        """
        pass

    @abstractmethod
    def set_target(self, target: float) -> None:
        """
        Set desired target acceleration.
        """
        pass

    @abstractmethod
    def advance(self, delta: float) -> None:
        """
        Advance the dynamic system by delta seconds.
        """
        pass

    @abstractmethod
    def timestamp(self):
        """
        Current time.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """
        Free all allocated resources.
        """
        pass
