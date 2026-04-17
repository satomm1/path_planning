from setuptools import setup
from catkin_pkg.python_setup import generate_distutils_setup

setup_args = generate_distutils_setup(
    packages=["path_planning"],
    package_dir={"": "src"},
    package_data={
        "path_planning": [
            "environments/grid_scenarios.json",
            "environments/*.yaml",
            "environments/*.pgm",
        ],
    },
)

setup(**setup_args)
