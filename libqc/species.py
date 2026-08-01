from dataclasses import dataclass

from scipy.constants import c, pi


@dataclass(frozen=True)
class Line:
    name: str
    wavelength: float
    linewidth: float
    weight: float

    @property
    def omega(self) -> float:
        """Angular frequency of the transition in rad/s."""
        return 2 * pi * c / self.wavelength


@dataclass(frozen=True)
class Atom:
    name: str
    mass_u: float
    lines: tuple[Line, ...]

    def get_line(self, name:str) -> Line:
        """Retrieves a line by name e.g. 'D1'."""
        for line in self.lines:
            if line.name == name:
                return line
        raise ValueError(f"Cannot find line {name}.")

K39 = Atom(
        name="K39",
        mass_u=38.96370668,
        lines=(
        Line("D1", 770.108385049e-9, 2 * pi * 5.956e6, 1 / 3),
        Line("D2", 766.700921822e-9, 2 * pi * 6.035e6, 2 / 3),
            )
        )

RB87 = Atom(
    name="Rb87",
    mass_u=86.909180520,
    lines=(
        Line("D1", 794.9789090e-9, 2 * pi * 5.7500e6, 1 / 3),
        Line("D2", 780.2412281e-9, 2 * pi * 6.0666e6, 2 / 3),
    ),
)

ATOMS: dict[str, Atom] = {
        "K39": K39,
        "Rb87": RB87,
        }
