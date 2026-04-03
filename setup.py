from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).resolve().parent
README = ROOT / "README.md"


setup(
    name="whitzard",
    version="0.1.0",
    description="Multimodal AIGC synthetic data generation framework MVP",
    long_description=README.read_text(encoding="utf-8") if README.exists() else "",
    long_description_content_type="text/markdown",
    python_requires=">=3.10",
    package_dir={"": "src"},
    packages=find_packages(where="src", include=["whitzard", "whitzard.*"]),
    include_package_data=True,
    install_requires=[
        "PyYAML>=6.0",
    ],
    entry_points={
        "console_scripts": [
            "whitzard=whitzard.cli.main:main",
        ]
    },
)
