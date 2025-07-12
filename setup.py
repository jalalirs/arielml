from setuptools import setup, find_packages

setup(
    name='arielml',
    version='0.1.0',
    description='A modular ML pipeline for astronomical data analysis',
    author='Your Name',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'lightgbm',
        'matplotlib',
        'seaborn',
        'jupyter',
        'astropy'
        # 'torch', 'tensorflow', # Uncomment as needed
    ],
) 