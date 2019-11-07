import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="medinify",
    version="0.0.1",
    install_requires=[
        'requests==2.22.0',
        'bs4==0.0.1',
        'tqdm==4.31.1',
        'pandas==0.25.0',
        'gensim==3.6.0',
        'spacy==2.2.0',
        'en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.0/en_core_web_sm-2.2.0.tar.gz',
        'sklearn==0.0',
        'torch==1.1.0',
        'torchtext==0.4.0',
        'pytest==4.3.0',
    ],
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
