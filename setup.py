import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="medinify",
    version="0.0.1",
    author="Example Author",
    author_email="author@example.com",
    description="For classifying medical text.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NanoNLP/medinify",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPLv3",
        "Operating System :: OS Independent",
    ],
)