from setuptools import setup, find_packages


test_deps = [
    "pytest>=6.2.3",
    "pytest-flask>=1.2.0",
    "pip>=21.0.1",
    "flake8>=3.9.2",
    "flake8-annotations>=2.6.2",
    "pytest-cov>=2.12.1",
    "black>=21.7b0"
]

deps = [
    "matplotlib>=3.4.3",
    "open_clip_torch",
    "pandas>=1.3.2",
    "scikit-learn>=0.24.2",
    "transformers>=4.39.2",
    "torch>=1.9.0",
    "torchvision>=0.10.0",
    "sglang[all]",
    "llava-torch",
    "ruff>=0.1.5"
]

serve_deps = [
    "dploy-kickstart>=0.1.5",
]


extras = {"test": test_deps + quality, "serve": serve_deps}

setup(
    name="caption-the-internet",
    version="0.1.0",
    author="Marianna Nezhurina",
    author_email="m.nezhurina@fz-juelich.de",
    description="Caption the images with variuous models",
    packages=find_packages(),
    install_requires=deps,
    tests_require=test_deps,
    extras_require=extras,
)
