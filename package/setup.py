from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = "0.0.1"
DESCRIPTION = "Package for Vegetable Phenotyping"
LONG_DESCRIPTION = "A package that allows to measure Vegetable phenotypes."

# Setting up
setup(
    name="vegphenome",
    version=VERSION,
    author="Asad Ismail",
    author_email="<asad.ismail@bayer.com>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=["opencv-python==4.1.1.26", "numpy", "colormath", "pandas", "Pillow", "matplotlib"],
    keywords=["python", "vegetables", "phenotyping"],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ],
)
