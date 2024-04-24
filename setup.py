from setuptools import setup, find_packages

setup(
    name='MVComp',
    version='0.9.4',
    author='NeuralABC',
    description='Multivariate Comparisons using Whole-brain and ROI-derived Metrics from MRI ',
    url='https://github.com/neuralabc/mvcomp',
    license='Apache License 2.0',
    py_modules=['mvcomp', 'plotting', 'utils'],
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