import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

__version__ = '0.0.2'

setuptools.setup(
    name='SDAQ',  # name="example-pkg-YOUR-USERNAME-HERE", # Replace with your own username
    version=__version__,
    author='Kilian Holzapfel',
    author_email='kilian.holzapfel@tum.de',
    description='A DAQ system based on hdf5 files',
    long_description=long_description,
    long_description_content_type="text/markdown",
    # url="https://github.com/pypa/sampleproject",
    # project_urls={
    #     "Bug Tracker": "https://github.com/pypa/sampleproject/issues",
    # },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)
