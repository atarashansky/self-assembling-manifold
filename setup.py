from setuptools import setup

setup(
    name='sam-algorithm',
    version='0.6.6',
    description='The Self-Assembling-Manifold algorithm',
    long_description="The Self-Assembling-Manifold algorithm for analyzing single-cell RNA sequencing data.",
    long_description_content_type='text/markdown',
    url='https://github.com/atarashansky/self-assembling-manifold',
    author='Alexander J. Tarashansky',
    author_email='tarashan@stanford.edu',
    keywords='scrnaseq analysis manifold reconstruction',
    py_modules=["SAM", "utilities", "SAMGUI"],
    install_requires=[
        'numpy',
        'scipy<=1.2.0',
        'pandas',
        'scikit-learn',
	'numba',
	'umap-learn', 'anndata<=0.6.19'],
    extras_require={
        'louvain': [
            'louvain', 'cython', 'python-igraph'],
        'leiden': [
            'leidenalg', 'cython', 'python-igraph'],
        'hdbscan': [
            'hdbscan'],
        'plot': [
            'ipythonwidgets', 'jupyter', 'plotly==4.0.0'],
        'scanpy': [
            'scanpy'],	    
        },
)
