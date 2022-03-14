# Documentation

This folder contains the scripts necessary to build Transformers4Rec's documentation.
You can view the generated [Transformers4Rec documentation here](https://nvidia-merlin.github.io/Transformers4Rec).

# Contributing to Docs

Follow the instructions below to be able to build the docs.

## Steps to follow:
1. In order to build the docs, we need to install Transformers4Rec in a conda env. [See installation instructions](https://github.com/NVIDIA-Merlin/Transformers4Rec).

2. Install required documentation tools and extensions:

```
cd Transformers4Rec
pip install -r requirements/dev.txt
```

3. If you have updated the docstrings, you need to delete the folder docs/source/api and then run this command within the docs/ folder
`sphinx-apidoc -f -o source/api ../transformers4rec`
`sphinx-apidoc -f -o source/api/merlin_standard_lib ../merlin_standard_lib`


4. Navigate to Transformers4Rec/docs/. If you have your documentation written and want to turn it into HTML, run makefile:

#be in the same directory as your Makefile

`make html`

This should run Sphinx in your shell, and outputs to build/html/index.html.

View docs web page by opening HTML in browser:
First navigate to /build/html/ folder, i.e., cd build/html and then run the following command:

`python -m http.server` or `python -m SimpleHTTPServer 8000`

Then, navigate a web browser to the IP address or hostname of the host machine at port 8000:

`https://<host IP-Address>:8000`

Now you can check if your docs edits formatted correctly, and read well.
## Decisions

### Source management: README and index files

* To preserve Sphinx's expectation that all source files are child files and directories
  of the `docs/source` directory, other content, such as the `notebooks` directory is
  copied to the source directory. You can determine which directories are copied by
  viewing `docs/source/conf.py` and looking for the `copydirs_additional_dirs` list.
  Directories are specified relative to the Sphinx source directory, `docs/source`.

* One consequence of the preceding bullet is that any change to the original files,
  such as adding or removing a topic, requires a similar change to the `docs/source/toc.yaml`
  file.  Updating the `docs/source/toc.yaml` file is not automatic.

* Because the GitHub browsing expectation is that a `README.md` file is rendered when you
  browse a directory, when a directory is copied, the `README.md` file is renamed to
  `index.md` to meet the HTML web server expectation of locating an `index.html` file
  in a directory.

* Add the file to the `docs/source/toc.yaml` file.  Keep in mind that notebooks are
  copied into the `docs/source/` directory, so the paths are relative to that location.
  Follow the pattern that is already established and you'll be fine.

### Adding links

TIP: When adding a link to a method or any heading that has underscores in it, repeat
the underscores in the link even though they are converted to hyphens in the HTML.

Refer to the following examples from HugeCTR:

* `../QAList.md#24-how-to-set-workspace_size_per_gpu_in_mb-and-slot_size_array`
* `./api/python_interface.md#save_params_to_files-method`

#### Docs-to-docs links

There is no concern for the GitHub browsing experience for files in the `docs/source/` directory.
You can use a relative path for the link.  For example, the following link is in the
`docs/source/hugectr_user_guide.md` file and links to the "Build HugeCTR from Source" heading
in the `docs/source/hugectr_contributor_guide.md` file:

```markdown
To build HugeCTR from scratch, refer to
[Build HugeCTR from source code](./hugectr_contributor_guide.md#build-hugectr-from-source).
```

#### Docs-to-repository links

To refer a reader to a README or program in a repository directory, state that
the link is to the repository:

```markdown
Refer to the sample Python programs in the
[examples/blah](https://github.com/NVIDIA-Merlin/Transformers4Rec/tree/main/examples/blah)
directory of the repository.
```

The idea is to let a reader know that following the link&mdash;whether from an HTML docs page or
from browsing GitHub&mdash;results in viewing our repository on GitHub.

> TIP: In the `release_notes.md` file, use the tag such as `v3.5` instead of `master` so that
> the link is durable.

#### Links to notebooks

The notebooks are published as documentation. The few exceptions are identified in the
`docs/source/conf.py` file in the `exclude_patterns` list:

```python
exclude_patterns = [
    # list RST, MD, and IPYNB files to ignore here
]
```

If the document that you link from is also published as docs, such as `release_notes.md`, then
a relative path works both in the HTML docs page and in the repository browsing experience:

```markdown
### Some awesome feature
    + ...snip...
    + ...snip...
    + Added the [awesome notebook](examples/awesome_notebook.ipynb) to show how to use the feature.
```

#### Links from notebooks to docs

Use a link to the HTML page like the following:

```markdown
<https://nvidia-merlin.github.io/Transformers4Rec/main/README.html>
```

> I'd like to change this in the future. My preference would be to use a relative
> path, but I need to research and change how Sphinx handles relative links.
