from setuptools import setup, find_packages

setup(
    name="pixelflow",
    version="0.1",
    author="Didrik Nielsen",
    author_email="didrik.nielsen@gmail.com",
    description="Code for paper 'Closing the Dequantization Gap: PixelCNN as a Single-Layer Flow'",
    long_description="",
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
