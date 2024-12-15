# Steel Industry Energy Consumption Optimization

## Overview
This project implements a deep learning solution for optimizing energy consumption in the steel manufacturing industry. Using PyTorch, we develop a neural network model to predict and control energy usage patterns, helping to reduce operational costs and improve energy efficiency.

## Features
- Data preprocessing and feature engineering for time-series energy data
- Deep learning model for energy consumption prediction
- Real-time monitoring capabilities

## Prerequisites
- Python 3.10 or higher
- Poetry for dependency management

## Installation

1. Install dependencies using Poetry:
```bash
poetry install
```

## Project Structure
```
steel_energy_optimization/
├── data/
│   ├── raw/                    # Raw dataset files
├── notebooks/                  # Jupyter notebooks for analysis
├── src/
│   └── steel_energy_opt/
│       ├── data_processor.py   # Data processing utilities
│       ├── model.py           # PyTorch model implementation
│       └── train.py           # Training script
├── poetry.lock               # Lock file for dependencies
├── pyproject.toml            # Project configuration
└── README.md                # This file
```

## Dataset
The project uses the "Steel Industry Energy Consumption Dataset" from the UCI Machine Learning Repository:
- Source: [Steel Industry Energy Consumption Dataset](https://archive.ics.uci.edu/static/public/851/steel+industry+energy+consumption.zip)
- Features include time-related data, energy consumption metrics, power factors, and load types
- 35,040 instances with data from 2018

## Usage

1. Download the dataset and place it in the `data/raw` directory:
```bash
mkdir -p data/raw
# Download the dataset from UCI ML Repository
```

2. Run the training script:
```bash
poetry run python -m steel_energy_opt.train
```

## Model Architecture
The neural network model consists of:
- Multiple fully connected layers with ReLU activation
- Dropout layers for regularization

## Acknowledgments
- UCI Machine Learning Repository for providing the dataset
- PyTorch team for the deep learning framework
- Contributors and maintainers of the project
