from setuptools import setup, find_packages


setup(
    name="crow",
    #version=get_version("crow/__init__.py"),
    author="The LSST DESC crow Contributors",
    license="BSD 3-Clause License",
    url="https://github.com/LSSTDESC/crow",
    packages=find_packages(),
    description="A comprehensive package for cluster theoretical prediction",
    long_description=open("README.md").read(),
    package_data={"": ["README.md", "LICENSE"]},
    include_package_data=True,
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: BSD 3-Clause",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ],
    #install_requires=["astropy>=4.0", "numpy", "scipy", "healpy"],
    #python_requires=">=" + str(required_py_version),
)
