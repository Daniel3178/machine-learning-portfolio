from setuptools import find_packages, setup

setup(
    name="ml_project_classifier",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    version="0.0.1",
    description="A movie review sentiment classifier project, authored by Daniel Ibrahimi"
)