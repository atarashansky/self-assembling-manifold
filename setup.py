from setuptools import setup

setup(

    name='sam-algorithm', 

    version='0.1.1',  

    description='The Self-Assembling-Manifold algorithm', 

    long_description="The Self-Assembling-Manifold algorithm for analyzing single-cell RNA sequencing data.",  

    long_description_content_type='text/markdown',  
    
    url='https://github.com/atarashansky/self-assembling-manifold',  

    author='Alexander J. Tarashansky',  

    author_email='tarashan@stanford.edu',  

    keywords='scrnaseq analysis manifold reconstruction',  

    py_modules=["source/SAM","source/utilities"],
    
    install_requires=['pandas','numpy','scikit-learn','matplotlib','scipy']
)
