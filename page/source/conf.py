from world_machine_page import html_theme_options, post_build, pre_build

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'World Machine'
copyright = '2025, Elton Cardoso do Nascimento, Paula Dornhofer Paro Costa'
author = 'Elton Cardoso do Nascimento, Paula Dornhofer Paro Costa'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    # "sphinx_mdinclude",
    "sphinx_favicon",
    "myst_parser",
    "nbsphinx",
    "ablog",
    'sphinx.ext.intersphinx'
]

templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
html_js_files = [
    ("custom-icons.js", {"defer": "defer"})
]
# source_suffix = ['.rst']


def skip(app, what, name, obj, would_skip, options):
    if name == "__init__":
        return False
    return would_skip


# MIST

myst_enable_extensions = [
    "colon_fence",
    "dollarmath",
    "html_image",
    "html_admonition",
    "html_image",
]

# Ablog

blog_path = "blog/index"
blog_title = "Blog"

myst_html_meta = {}

# Build


def setup(app):

    app.config.m2r_parse_relative_links = True
    app.connect("build-finished", post_build)
    # app.connect("autodoc-skip-member", skip)


pre_build()
