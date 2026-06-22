from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

setup_args = generate_distutils_setup(
    packages=["social_path_planning"],
    package_dir={"": "src"},
    package_data={
        "social_path_planning": [
            "environments/grid_scenarios.json",
            "environments/*.yaml",
            "environments/*.pgm",
        ],
    },
)

setup(**setup_args)
