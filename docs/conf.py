# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
sys.path.insert(0, os.path.abspath('.'))

project = 'Algosto'
copyright = '2024 Algosto'
author = 'Melvine Nargeot'
release = '0.0.18'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx-prompt',
    'sphinx.ext.mathjax',
    'sphinx.ext.autodoc',
    'sphinx_design',
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

html_static_path = ['images']

html_logo = "images/logo.png"
html_favicon = "images/favicon.png"
html_title = "Algosto Docs"

autosummary_generate = False
numpydoc_show_class_members = False
