from setuptools import setup, find_packages

setup(
    name='MVComp',
    version='0.9.3',
    author='NeuralABC',
    description='Multivariate Comparisons using Whole-brain and ROI-derived Metrics from MRI ',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'seaborn',
        'matplotlib',
        'scikit-learn',
        'nibabel',
        'nilearn',
    ]

)