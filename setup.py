from setuptools import setup, find_packages

setup(
    name="kinetics",
    version="0.1.0",
    description="Binding model fitting for absorbance data (2- and 3-species models)",
    author="Rajka Pejanovic",
    author_email="rajka.pejanovic@ens.psl.eu",
    packages=find_packages(),  # automatically finds 'kinetics', 'kinetics.utils', etc.
    include_package_data=True,
    install_requires=[
        "lmfit==1.3.2",
        "numpy==1.26.4",
        "pandas==2.2.3",
        "matplotlib==3.9.2",
        "openpyxl==3.1.5",
        "sympy==1.13.3"
    ],
    entry_points={
        "console_scripts": [
            "kinetics-fit=kinetics.run_fit:main"
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
