from setuptools import find_packages, setup

setup(
    name="rtnls_fundusprep",
    version="0.1.0",
    author="Bart Liefers",
    author_email="your.email@example.com",
    description="A brief description of package1",
    packages=find_packages(),
    install_requires=[
        "numpy == 1.*",
        "pandas == 2.*",
        "scikit-learn==1.*",
        "scikit-image==0.24.0",
        "opencv-python == 4.*",
        "matplotlib==3.*",
        "joblib==1.*",
        "tqdm == 4.*",
        "Pillow == 9.*",
        "click==8.*",
    ],
    package_data={},
)
