from setuptools import setup

setup(
    name='sam-algorithm',
    version='0.1.11',
    description='The Self-Assembling-Manifold algorithm',
    long_description="The Self-Assembling-Manifold algorithm for analyzing single-cell RNA sequencing data.",
    long_description_content_type='text/markdown',
    url='https://github.com/atarashansky/self-assembling-manifold',
    author='Alexander J. Tarashansky',
    author_email='tarashan@stanford.edu',
    keywords='scrnaseq analysis manifold reconstruction',
    py_modules=["SAM", "utilities"],
    install_requires=[
        'numpy>=1.15',
        'scipy',
        'pandas',
        'scikit-learn>=0.20'],
    extras_require={
        'louvain': [
            'louvain',
            'scanpy',
            'anndata'],
        'umap': [
            'numba>=0.37,<0.40',
            'umap-learn'],
        'plot': [
            'matplotlib'],
        },
)
