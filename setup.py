from setuptools import setup


with open('deal_or_no_deal/_version.py') as version_file:
    exec(version_file.read())

with open('README.md') as r:
    readme = r.read()

setup(
    name='deal_or_no_deal',
    version=__version__,
    description='Playing Deal or No Deal better than a human.',
    long_description=readme,
    install_requires=[
        'fire',
        'gym',
        'joblib',
        'Keras',
        'numpy',
        'pandas',
        'sklearn',
        'tensorflow',
        'tqdm',
        'xgboost',
        'xlrd',
    ]
)
