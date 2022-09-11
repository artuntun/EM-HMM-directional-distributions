# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

long_description = None
INSTALL_REQUIRES = [
    'scipy==1.9.1',
    'pandas==1.4.4',
]
EXTRAS_REQUIRE = {
    'local': [
        'matplotlib==3.5',
        'mypy==0.950',
    ],
}

setup_kwargs = {
    'name': 'em-hmm-directional',
    'version': '0.0.1',
    'description': 'Expectiation Maximization for von Mises-Fisher and Watson Distributions',
    'long_description': long_description,
    'license': 'MIT',
    'author': 'Arturo Arranz',
    # 'url': 'https://github.com/uizard-io/pip-package-uizard-nolybab',
    'packages': find_packages(),
    'package_data': {
        'em_hmm_directional': ['py.typed'],
    },
    'zip_safe': False,
    'install_requires': INSTALL_REQUIRES,
    'extras_require': EXTRAS_REQUIRE,
    'python_requires': '>=3.7',
}


setup(**setup_kwargs)
