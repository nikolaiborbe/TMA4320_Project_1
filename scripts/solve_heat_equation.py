import numpy as np

from project import (
    solve_heat_equation,
    load_config,
    Config
)

config: Config = load_config()

solve_heat_equation(config)