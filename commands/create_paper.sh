#!/bin/bash

cd paper

sudo xelatex --shell-escape main
sudo bibtex main
sudo xelatex --shell-escape main
sudo xelatex --shell-escape main

cd ..