from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = ["keras~=2.7", "matplotlib~=3.5", "numpy>=1.2", "tensorflow>=2.5", "sklearn"]

setup(
    name="oparch",
    version="0.0.5",
    author="Ilmari Vahteristo",
    author_email="i.vahteristo@gmail.com",
    description="first package",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/ilmari99/oparch/",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
)