# Hybrid Spiking Neural Network - Transformer Video Classification Model

This is the code implementation of the SSN architecture of my BSc thesis in computer science available here.

HyTSSN is a Python-based project designed for simulating and analyzing hybrid temporal-spatial spiking neural networks. The project provides modular components for building, simulating, and visualizing spiking neural network architectures.

---

## Features
- Modular neural network components including synapses, dendrites, and currents.
- Temporal and spatial resolution handling for simulations.
- Customizable encoding mechanisms.
- Visualization support for model architectures.

---

## Project Structure

### Source Code Files
- **`Currents.py`**: Defines and manages the neural currents for the model.
- **`Dendrite.py`**: Implements dendrite functionality in the network.
- **`Encoder.py`**: Handles input encoding for network simulations.
- **`Input.py`**: Manages the input data structures and preprocessing.
- **`Models.py`**: Core implementation of the spiking neural network models.
- **`Synapse.py`**: Contains logic for synapse interactions and updates.
- **`TimeResolution.py`**: Manages the time resolution of network simulations.
- **`main.py`**: The main entry point for running simulations.

### Additional Files
- **`Model Architecture Design.png`**: A diagram illustrating the architecture of the model.
- **`README.md`**: Project documentation (this file).

---

## Prerequisites
- Python 3.10 or higher.
- Required Python libraries:
  - (List libraries here, e.g., NumPy, Matplotlib, etc., if applicable).

---

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/HyTSSN.git
   ```
2. Navigate to the project directory:
   ```bash
   cd HyTSSN
   ```
3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage
1. To run the main simulation:
   ```bash
   python main.py
   ```
2. Customize model parameters in the respective source files (`Models.py`, `Encoder.py`, etc.).

---

## Architecture Overview
The **Hybrid Temporal-Spatial Spiking Neural Network** architecture is composed of:
- **Input Layer**: Encodes input stimuli using the `Encoder.py` module.
- **Hidden Layers**: Simulates synaptic interactions (`Synapse.py`) and dendritic processing (`Dendrite.py`).
- **Output Layer**: Aggregates results to provide final outputs.

<p align="center">
  <img src="https://github.com/TheRNB/HyTSSN/blob/main/Model%20Architecture%20Design.png" width="510">
</p>

---

## Contribution
Contributions are welcome! Please fork the repository and submit a pull request with your changes.

---

## License
This project is licensed under the MIT License.

---

## Contact
For questions or support, reach out to:
- Author: Aaron Bateni
- Email: aaron.bateni@ut.ac.ir
