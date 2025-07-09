
from setuptools import setup, find_packages

setup(
    name='bbc_news_nlp_pipeline',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'transformers==4.40.1',
        'datasets==2.19.1',
        'scikit-learn==1.4.1',
        'spacy==3.7.2',
        'nlpaug==1.1.11',
        'evaluate==0.4.1',
        'torch>=2.0.0',
        'pandas>=2.0.0',
        'matplotlib>=3.5.0',
        'seaborn>=0.11.2',
    ],
    author='Your Name',
    description='NLP pipeline for BBC news classification, summarisation, and bias detection',
    keywords='NLP transformers news classification summarisation spacy bias',
)
