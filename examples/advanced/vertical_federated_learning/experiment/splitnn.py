import os

from nvflare import SimulatorRunner

import logging


logging.getLogger().setLevel(logging.DEBUG)

if __name__ == "__main__":
    simulator = SimulatorRunner(
        job_folder=f"jobs/splitnn",
        workspace="/tmp/nvflare/splitnn",
        n_clients=2,
        threads=2
    )
    run_status = simulator.run()
    print("Simulator finished with run_status", run_status)
