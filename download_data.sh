#!/bin/bash
set -e  # exit on error

# Create directories
mkdir -p data
mkdir -p data/raw

gdown --id 0B7EVK8r0v71pa2EyNEJ0dE9zbU0 -O data/raw/img.zip
gdown --folder https://drive.google.com/drive/folders/19J-FY5NY7s91SiHpQQBo2ad3xjIB42iN -O data/raw/ --remaining-ok
