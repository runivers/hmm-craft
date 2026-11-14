import hmmcraft

needs_sphinx = '2.0'
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx_gallery.gen_gallery',
]

project = 'hmm-craft'
copyright = ('2010-present, hmmlearn developers; '
             '2026-present, hmm-craft developers (BSD License)')
version = release = hmmcraft.__version__
default_role = 'py:obj'
pygments_style = 'sphinx'
language = 'en'

# -- Options for extensions --------------------------------------------------

autodoc_default_options = {
    'members': None,
    'inherited-members': None,
    'special-members': '__init__',
}

intersphinx_mapping = {
    'sklearn': ('https://scikit-learn.org/stable/', None),
}

napoleon_use_ivar = True
napoleon_use_rtype = False

sphinx_gallery_conf = {
    'examples_dirs': '../../examples',
    'gallery_dirs': 'auto_examples'
}

# -- Options for HTML output -------------------------------------------------

html_theme = 'pydata_sphinx_theme'
html_theme_options = {
    'github_url': 'https://github.com/hmmlearn/hmmlearn',
}
html_css_files = ['hide_some_gallery_elements.css']
html_static_path = ['_static']
htmlhelp_basename = 'hmmcraft_doc'

# -- Options for LaTeX output ------------------------------------------------

latex_documents = [('index', 'user_guide.tex', 'hmm-craft user guide',
                    'hmmlearn and hmm-craft developers', 'manual'), ]
