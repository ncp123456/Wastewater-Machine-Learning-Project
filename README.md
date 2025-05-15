# Docker MAIA Project

A machine learning project for analyzing and predicting wastewater treatment outcomes using various ML models. This project aims to enhance the efficiency and effectiveness of wastewater treatment processes through data-driven insights and predictions.

## Project Overview

This project implements various machine learning models for time series forecasting of wastewater treatment parameters, including:
- LSTM (Long Short-Term Memory)
- RNN (Recurrent Neural Network)
- Random Forest
- Linear Regression
- XGBoost
- TabPFN
- Lag-Llama

The models analyze various treatment parameters such as:
- EFF_OP (Operational Efficiency)
- EFF_TSS (Total Suspended Solids Efficiency)
- EFF_NHX (Ammonia Efficiency)
- EFF_TP (Total Phosphorus Efficiency)

## Prerequisites

- Docker
- NVIDIA GPU with CUDA support (for GPU acceleration)
- NVIDIA Container Toolkit
- Git

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ncp123456/Wastewater-Machine-Learning-Project.git
cd Wastewater-Machine-Learning-Project
```

2. Build the Docker image:
```bash
docker build --build-arg HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN} -t maia-project .
```

## Project Structure

```
docker-maia-project/
├── src/                    # Source code directory
│   ├── file_processing.py  # Data processing utilities
│   ├── lstm_model.py       # LSTM model implementation
│   ├── rnn_model.py        # RNN model implementation
│   ├── random_forest.py    # Random Forest implementation
│   ├── linear_regression.py # Linear Regression implementation
│   ├── xgboost_model.py    # XGBoost implementation
│   ├── tabpfn_model.py     # TabPFN implementation
│   ├── lag_llama_model.py  # Lag-Llama implementation
│   ├── utils.py           # Utility functions
│   ├── config.py          # Configuration settings
│   └── plotting.py        # Visualization utilities
├── models/                # Directory for model checkpoints
├── output/               # Directory for model outputs
├── plots/               # Directory for generated plots
├── Dockerfile           # Docker configuration
├── requirements_base.txt # Base Python dependencies
└── requirements_more.txt # Additional ML framework dependencies
```

## Usage

1. Run a specific model using Docker:
```bash
docker run --gpus all \
    -v ${PWD}/src:/app/src \
    -v ${PWD}/models:/app/models \
    -v ${PWD}/output:/app/output \
    -v ${PWD}/plots:/app/plots \
    -e HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN} \
    maia-project src/<model_name>.py
```

Replace `<model_name>` with one of:
- lstm_model
- rnn_model
- random_forest
- linear_regression
- xgboost_model
- tabpfn_model
- lag_llama_model

## Environment Variables

- `HUGGINGFACE_TOKEN`: Required for downloading model weights from HuggingFace
- `MODEL_CHECKPOINT_DIR`: Directory for model checkpoints (default: /app/models)
- `OUTPUT_DIR`: Directory for model outputs (default: /app/output)
- `PLOTS_DIR`: Directory for generated plots (default: /app/plots)

## Dependencies

The project uses two requirements files:
- `requirements_base.txt`: Core dependencies including pandas, numpy, and ML frameworks
- `requirements_more.txt`: Additional ML framework dependencies

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License

Copyright (c) 2024 [Nathan Peot]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Contact

Nathan Peot
- GitHub: [ncp123456](https://github.com/ncp123456)
- Email: [Your email address]

## Data Requirements

The project requires two main data files:
1. `facility (4).csv`: Contains facility measurement data
2. `sumo1b (2).csv`: Contains simulation data

These files should be placed in the `src/` directory before running the models.
