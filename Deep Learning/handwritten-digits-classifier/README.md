# MNIST Handwritten Digits Classifier

This project contains a basic deep learning model to classify handwritten digits from the MNIST dataset. The model is trained to recognize digits (0-9) and predict the digit in new images with reasonable accuracy.

## Project Structure

- **MNIST_Handwritten_Digits-STARTER.ipynb**: Jupyter Notebook containing the code for loading, training, and evaluating the model on the MNIST dataset.
- **mnist_model.pth**: Pretrained model file saved in PyTorch format. This file contains the trained weights of the classifier.
- **requirements.txt**: Lists the required Python packages to run the notebook and train/evaluate the model.

## Requirements

To install the dependencies, use:
```bash
pip install -r requirements.txt
```

## Usage

1. **Run the Notebook**:  
   Open the `MNIST_Handwritten_Digits-STARTER.ipynb` file in Jupyter Notebook or JupyterLab and execute the cells to load the model and make predictions.

2. **Pretrained Model**:  
   The `mnist_model.pth` file contains the pretrained weights. You can load this model in PyTorch to evaluate it on new data or for further training.

## About the Model

The model was trained on the MNIST dataset, which consists of 28x28 pixel grayscale images of handwritten digits. The classifier is based on a simple neural network architecture that achieves reasonable accuracy on this dataset.

## License

This project is open-source. Feel free to use, modify, and distribute as needed.
