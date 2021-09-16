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

