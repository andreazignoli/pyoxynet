import setuptools

setuptools.setup(
    name="pyoxynet",
    version="0.0.0.4",
    author="Andrea Zignoli",
    author_email="andrea.zignoli@unitn.it",
    description="Python package of the Oxynet project (visit www.oxynet.net)",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
    package_data={'': ['models/*']},
    #exclude_package_data={
    #    '': 'debugging.py.c'},
    python_requires='>=3.8',
)