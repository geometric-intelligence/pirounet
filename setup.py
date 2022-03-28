"""Create instructions to build the move package."""

import setuptools

requirements = []

setuptools.setup(
    name="move",
    maintainer="Mathilde Papillon",
    version="0.0.1",
    maintainer_email="papillon@ucsb.edu",
    description="Dancing Geometries",
    long_description=open("README.md", encoding="utf8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/bioshape-lab/move.git",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    zip_safe=False,
)
