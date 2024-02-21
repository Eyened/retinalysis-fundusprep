from setuptools import setup, find_packages

setup(
    name="rtnls_fundusprep",
    version="0.1.0",
    author="Bart Liefers",
    author_email="your.email@example.com",
    description="A brief description of package1",
    packages=find_packages(),
    install_requires=[
        "numpy == 1.*",
        "pandas == 1.*",
        "scikit-learn==1.*",
        "Pillow == 9.*",
        "opencv-python == 4.*",
        "scikit-image==0.21.0",
        "pydicom==2.3.1",
    ],
    package_data={},
)
