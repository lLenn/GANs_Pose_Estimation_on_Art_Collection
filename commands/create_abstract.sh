#!/bin/bash

cd paper/abstract

sudo pdflatex --shell-escape abstract
sudo bibtex abstract
sudo makeglossaries abstract
sudo pdflatex --shell-escape abstract
sudo pdflatex --shell-escape abstract

cd ../../