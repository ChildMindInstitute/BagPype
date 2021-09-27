import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="BagPype",                     # This is the name of the package
    version="0.0.4",                        # The initial release version
    author="Jacob DeRosa",                     # Full name of the author
    author_email = "jacob.derosa@colorado.edu",
    long_description_content_type = "text/markdown",
    url = "https://github.com/ChildMindInstitute/BagPype",
    description="Bagging Enhanced Network Analysis",
    long_description=long_description,      # Long description read from the the readme file
    packages=setuptools.find_packages(),    # List of all python modules to be installed
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],                                      # Information to filter the project on PyPi website
    python_requires='>=3.6',                # Minimum version requirement of the package
    package_dir={"": "scripts"},
    packages=setuptools.find_packages(where="scripts"),
    install_requires=[]                     # Install other dependencies if any
)


