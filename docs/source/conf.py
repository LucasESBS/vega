# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import sphinx_rtd_theme
#import vega

sys.path.insert(0, os.path.abspath('../..'))
sys.setrecursionlimit(1500)


# -- Project information -----------------------------------------------------

project = 'VEGA'
copyright = '2021, Lucas Seninge, Ioannis Anastopoulos'
author = 'Lucas Seninge, Ioannis Anastopoulos'

# The full version, including alpha/beta/rc tags
#version = vega.__version__
#release = vega.__version__
version = '0.0.2'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
                'sphinx.ext.todo', 
                'sphinx.ext.viewcode',
                'sphinx.ext.autodoc',
                'nbsphinx',
                'nbsphinx_link',
                'sphinx_rtd_theme',
                'sphinx.ext.napoleon',
                'sphinx.ext.intersphinx',
                'sphinx_design',
                'sphinx_autodoc_typehints',
                'sphinx.ext.autosummary',]
# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# nbsphinx specific settings
exclude_patterns = ["_build", "**.ipynb_checkpoints"]
nbsphinx_execute = "never"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
#html_theme = 'alabaster'
#html_theme = "sphinx_rtd_theme"

html_theme = 'furo'
html_title = "VEGA"
html_logo = "_static/logo.png"

html_theme_options = {
    "sidebar_hide_name": True,
    "light_css_variables": {
        "color-brand-primary": "#003262",
        "color-brand-content": "#003262",
        "admonition-font-size": "var(--font-size-normal)",
        "admonition-title-font-size": "var(--font-size-normal)",
        "code-font-size": "var(--font-size--small)",
    },
}



# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


latex_elements = {
    'papersize':'letterpaper',
    'pointsize':'10pt',
    'preamble':'',
    'figure_align':'htbp'
}

# Sorting of methods
autodoc_member_order = 'bysource'
# Add init to docs
autoclass_content = 'both'

# Napoleon settings
autodoc_member_order = "bysource"
autosummary_generate = True
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None
napoleon_attr_annotations = True
napoleon_custom_sections = [("Params", "Parameters")]
todo_include_todos = False
numpydoc_show_class_members = False
annotate_defaults = True

