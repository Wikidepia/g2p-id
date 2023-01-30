from setuptools import find_packages, setup

setup(
    name="g2p-id",
    packages=find_packages(exclude=[]),
    version="0.0.8",
    license="MIT",
    description="Indonesian Grapheme-to-Phoneme (G2P)",
    author="Akmal",
    author_email="akmal@depia.wiki",
    long_description_content_type="text/markdown",
    url="https://github.com/Wikidepia/g2p-id",
    install_requires=[
        "sacremoses>=0.0.41",
        "nltk>=3.7",
        "onnxruntime>=1.7.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
    ],
    include_package_data=True,
)
