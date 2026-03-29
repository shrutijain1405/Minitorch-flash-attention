from setuptools import setup, find_packages

def parse_requirements(filename):
    with open(filename, "r") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="minitorch",
    version="0.0",
    packages=find_packages(),
    install_requires=parse_requirements("requirements.txt"),
)