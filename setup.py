from setuptools import setup

setup(
    name='sam-algorithm',
    version='0.4.4',
    description='The Self-Assembling-Manifold algorithm',
    long_description="The Self-Assembling-Manifold algorithm for analyzing single-cell RNA sequencing data.",
    long_description_content_type='text/markdown',
    url='https://github.com/atarashansky/self-assembling-manifold',
    author='Alexander J. Tarashansky',
    author_email='tarashan@stanford.edu',
    keywords='scrnaseq analysis manifold reconstruction',
    py_modules=["SAM", "utilities"],
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'scikit-learn',
	'numba<0.43.0',
	'umap-learn', 'anndata'],
    extras_require={
        'louvain': [
            'louvain'],
        'hdbscan': [
            'hdbscan'],
        'plot': [
            'matplotlib'],
        'scanpy': [
            'scanpy'],	    
        },
)
