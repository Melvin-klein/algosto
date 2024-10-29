from setuptools import setup, find_packages

setup(
    name="algosto",
    version="0.0.6",
    author="Melvine Nargeot",
    author_email="melvine.nargeot@gmail.com",
    description="""Algosto is a package that
    includes stochastic optimization algorithms.""",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Melvin-klein/algosto",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[],
)
