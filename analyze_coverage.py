"""兼容入口：实现已移到 active_recon.experiment.coverage_analysis。命令与参数保持不变。"""
from active_recon.experiment.coverage_analysis import *  # noqa: F401,F403
from active_recon.experiment.coverage_analysis import main  # noqa: F401

if __name__ == "__main__":
    main()
