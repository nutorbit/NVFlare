import os
from nvflare import SimulatorRunner


if __name__ == "__main__":
    simulator = SimulatorRunner(
        job_folder=f"jobs/psi",
        workspace="/tmp/nvflare/vertical_xgb_psi",
        n_clients=2,
        threads=2
    )
    run_status = simulator.run()
    print("Simulator finished with run_status", run_status)
