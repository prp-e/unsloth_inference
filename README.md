# Unsloth Inference Guide

Since I am going to fine tune different models and mostly locally, I preferred a fast and optimized way. `unsloth` is the best way possible, but most of the codes on the internet are very specialized towards the model they try to fine tune. 

Then, I decided to make a code base where people with a little bit of change in the code base, be able to at least do the inference phase. So in this code this is possible. 

## Important notes for running this repository

- To my knowledge, Python 3.11+ is the best way of running unsloth. 
- If you want another model for inference, just change `model_name` in `main.py` code manually. I personally prefer what unsloth share on [their huggingface account](https://hf.co/unsloth) and if you're going to load them in 4 bits, go for the ones with `bnb-4bit` in the name. 
- Unsloth is relying on `triton` package and it only works with Linux. So if you are like me and want to run the code on a windows machine, just use WSL.

## The hardware

I ran the code on my Personal laptop (LOQ) with a 2050 GPU. 