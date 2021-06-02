import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


required = ['torch==1.5.1',
            'torchvision==0.6.1',
            'numpy==1.17',
            'pandas',
            'sklearn',
            'networkx',
            'seaborn',
            'scipy==1.4',
            'scanpy==1.5.1',
            'umap-learn',
            'adjustText',
            'scvi-tools==0.9.0']

setuptools.setup(
    name="VEGA-LucaSC",
    version="0.0.2",
    author="Lucas Seninge",
    author_email="lseninge@ucsc.edu",
    description="Package for VEGA: VAE Enhanced by Gene Annotations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/LucasESBS/vega",
    packages=setuptools.find_packages(),
    install_requires=required,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
