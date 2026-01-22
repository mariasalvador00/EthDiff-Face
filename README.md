# EthDiff-Face

EthDiff-Face is a research project focused on **ethnicity-aware synthetic face generation using diffusion models**.  
The goal of this work is to generate high-quality, demographically controlled synthetic face data that can be used to improve fairness, generalisation, and robustness in face recognition systems.


## Project Structure

Here, we present the scripts used throughout this work. If you want to access the trained models and the encoded synthetic data, you can do so through https://drive.google.com/drive/folders/1h0qUyOW-AVu9t2rPhj4KUGxznMw4QD_L?usp=sharing



## Usage

The training script for each of the ethnicity specific IDiff-Face models is under main.py and the sampling script is under sample.py

After sampling, the data is encoded using encode.py and the genuine-imposter pairs are generated with evaluation/generate_pairs.py



