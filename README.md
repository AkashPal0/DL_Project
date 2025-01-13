# AutoGaitFormer: Gait Recognition Using Periodic Gait Cycles

## Overview
AutoGaitFormer is a cutting-edge deep learning model designed for gait recognition. The model leverages the periodic nature of human gait cycles to achieve state-of-the-art performance in identifying individuals based on their walking patterns. This project demonstrates how deep learning can effectively model and interpret periodic signals in gait recognition tasks, providing robust and accurate results.

## Features
- **Periodic Gait Cycle Detection**: Identifies and utilizes periodicity in gait patterns to enhance recognition accuracy.
- **Transformer Architecture**: Employs a transformer-based architecture for efficient feature extraction and temporal modeling.
- **High Performance**: Achieves superior accuracy on standard gait recognition datasets.
- **Scalable and Modular**: Designed to be easily adaptable for various datasets and applications.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/AkashPal0/DL_Project.git
   cd DL_Project
   ```
2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset Preparation
1. Download the gait recognition dataset of your choice (e.g., CASIA-B, OU-ISIR).
2. Preprocess the dataset to extract frames or silhouettes, ensuring compatibility with the model.
3. Update the `config.yaml` file to include the dataset path and preprocessing parameters.

## Usage
### Training
To train the AutoGaitFormer model:
```bash
python train.py --config config.yaml
```
### Evaluation
To evaluate the model on a test dataset:
```bash
python evaluate.py --config config.yaml
```
### Inference
To perform inference on new gait data:
```bash
python infer.py --input <path_to_input_data>
```

## Configuration
Modify the `config.yaml` file to customize the training and evaluation settings:
- **Dataset Parameters**: Paths, preprocessing methods, and augmentation settings.
- **Model Parameters**: Transformer architecture details, input dimensions, and hyperparameters.
- **Training Parameters**: Learning rate, batch size, number of epochs, and optimizer settings.


## Repository Structure
```
DL_Project/
├── data/                # Dataset and preprocessing scripts
├── models/              # Model architecture and utilities
├── experiments/         # Training and evaluation scripts
├── configs/             # Configuration files
├── utils/               # Helper functions and modules
├── requirements.txt     # Python dependencies
├── train.py             # Training script
├── evaluate.py          # Evaluation script
├── infer.py             # Inference script
└── README.md            # Project documentation
```

## Future Work
- Extending support for 3D gait recognition.
- Incorporating real-time inference capabilities.
- Experimenting with hybrid architectures combining CNNs and transformers.

## Contributions
Contributions are welcome! Feel free to open issues or submit pull requests to enhance the project.

## Acknowledgments
We thank the creators of the datasets and foundational libraries used in this project. Special appreciation goes to the research community for inspiring the development of OpenGait code base.

