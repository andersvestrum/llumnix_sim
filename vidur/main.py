from vidur.config import SimulationConfig
from vidur.simulator import Simulator
from vidur.utils.random import set_seeds
import plotly.io as pio


def main() -> None:
    config: SimulationConfig = SimulationConfig.create_from_cli_args()

    set_seeds(config.seed)

    pio.renderers.default = "browser"

    simulator = Simulator(config)
    simulator.run()
    simulator._write_output()


if __name__ == "__main__":
    main()
