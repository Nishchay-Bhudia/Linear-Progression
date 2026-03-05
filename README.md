# Linear Regression 

---

## Script Functionality

The program follows a standard supervised learning workflow:

1. **Data Collection**: The user inputs raw `X` and `Y` coordinates via the console.  
2. **Preprocessing**: Input strings are parsed and converted into NumPy float arrays.  
3. **Optimisation**: The script runs a training loop for 1,000 iterations (epochs) to find the line of best fit.  
4. **Inference**: Once trained, the model prompts the user for a new `X` value to predict an unknown `Y` value.  
5. **Evaluation**: Two charts are generated:
   - Regression line against the data  
   - Error reduction (Loss) over time  

---

## The Mathematics

The model aims to solve the linear equation:

$$
y = wx + b
$$

Where:  
- \(w\) is the **Weight (Slope)**  
- \(b\) is the **Bias (Y-intercept)**  

### 1. Cost Function (Mean Squared Error)

The accuracy of the model is measured using **Mean Squared Error (MSE)**:

$$
MSE = \frac{1}{n} \sum_{i=1}^{n} (y_{pred, i} - y_i)^2
$$

### 2. Gradient Descent

Optimisation is achieved by calculating the gradients (partial derivatives) of the MSE with respect to \(w\) and \(b\):

$$
dw = \frac{-2}{n} \sum x(y - y_{pred})
$$

$$
db = \frac{-2}{n} \sum (y - y_{pred})
$$

The parameters are updated using a **Learning Rate** :
$$
w = w - (\alpha \cdot dw)
$$

$$
b = b - (\alpha \cdot db)
$$

---

## Technical Requirements

Ensure you have the following Python packages installed:

- `numpy`  
- `matplotlib`  

Install them via pip:

```bash
pip install numpy matplotlib
```

---

## Setup and Configuration

- **Learning Rate**: Set to `0.000001`.  
  - If the training loss does not decrease, this value may be too small.  
  - If the loss becomes `NaN`, the value is likely too high.  
- **Epochs**: Set to `1000`. Controls how many times the model sees the entire dataset during training.  

---

## Usage

1. Run the script:

```bash
git clone https://github.com/Nishchay-Bhudia/Linear-Progression.git
```

2. Enter your `X` and `Y` data when prompted.  
3. After training, input a new `X` value to get a predicted `Y`.  
4. Visualisations will be displayed showing:
   - The regression line  
   - Loss reduction over epochs  

---

## Purpose

This project is designed for **educational purposes**, helping students and beginners understand the inner workings of Linear Regression and Gradient Descent without relying on high-level libraries.
