from setuptools import setup, find_packages

setup(
    name='sam-algorithm',
    version='0.7.0',
    description='The Self-Assembling-Manifold algorithm',
    long_description="The Self-Assembling-Manifold algorithm for analyzing single-cell RNA sequencing data.",
    long_description_content_type='text/markdown',
    url='https://github.com/atarashansky/self-assembling-manifold',
    author='Alexander J. Tarashansky',
    author_email='tarashan@stanford.edu',
    keywords='scrnaseq analysis manifold reconstruction',
    python_requires='>=3.6',
    #py_modules=["SAM", "utilities", "SAMGUI"],
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'scikit-learn','packaging',
	'numba',
	'umap-learn', 'anndata>=0.7rc1'],
    extras_require={
        'louvain': [
            'louvain', 'cython', 'python-igraph'],
        'leiden': [
            'leidenalg', 'cython', 'python-igraph'],
        'hdbscan': [
            'hdbscan'],
        'plot': [
            'ipythonwidgets', 'jupyter', 'plotly==4.0.0', 'matplotlib'],
        'scanpy': [
            'scanpy'],	    
        },
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False
)
