import os
import sys
# 1 – Project root
sys.path.insert(0, os.path.abspath('..'))

# 2 – Extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx_autodoc_typehints',
]

# 3 – Theme
html_theme = 'sphinx_rtd_theme'

# 4 – default opions
autodoc_default_options = {
    'members': True,            # incluye métodos y atributos
    'undoc-members': True,      # muestra miembros sin docstring (para que veas todo)
    'show-inheritance': True,
    'inherited-members': True,
}

# simulate importations
autodoc_mock_imports = [
    "scipy",
    "plotly",
    "kaleido",
    "matplotlib",
    "IPython",
    "pytest",
    "graphviz",
    # numpy must be installed to use autodoc
    # pandas must be installed to use autodoc
]
