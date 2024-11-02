# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Algosto'
copyright = '2024, Melvine Nargeot'
author = 'Melvine Nargeot'
release = '0.0.12'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx-prompt',
    'sphinx.ext.mathjax',
    'sphinx.ext.autodoc',
    'sphinx_design',
    'sphinx.ext.autosummary',
    'numpydoc',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_theme_options = {
    "repository_url": "https://github.com/Melvin-klein/algosto",
    "use_repository_button": True,
    "use_fullscreen_button": False,
    "pygments_light_style": "colorful",
    "pygments_dark_style": "gruvbox-dark",
}

html_static_path = ['_static']

html_logo = "_static/logo.png"
html_favicon = "_static/favivon.png"
html_title = "Algosto Docs"

autosummary_generate = True
