import numpy as np
import matplotlib.pyplot as plt #needed for graphical representation

#input data

print("Enter your X values separated by commas (example: 1,2,3,4)")
x_input = input("X values: ")

print("Enter your Y values separated by commas (example: 2,4,6,8)")
y_input = input("Y values: ")

#convert input into numpy arrays
x = np.array([float(i) for i in x_input.split(",")])  # uses split to turn array into indavdiaual values
y = np.array([float(i) for i in y_input.split(",")])



#initialize parameters
w = 0.0
b = 0.0
learning_rate = 0.000001    # how big of a step we take during each iteration
epochs = 1000
n = len(x)

loss_history = []

#loop - training the model

for epoch in range(epochs):

    y_pred = w * x + b #predicted y values formula

    loss = np.mean((y_pred - y) ** 2)
    loss_history.append(loss)

    dw = (-2/n) * np.sum(x * (y - y_pred)) # formula for slope
    db = (-2/n) * np.sum(y - y_pred) # formula for intercept

    w -= learning_rate * dw #formula for slope update
    b -= learning_rate * db #formula for intercept update

print("Training complete!")
print("Learned slope (w):", w)
print("Learned intercept (b):", b)

#preditction

new_x = float(input("Enter a new X value to predict Y: "))
prediction = w * new_x + b

print("Predicted Y value:", prediction)

#mathplotlib visualization

y_pred_line = w * x + b # predicted line

plt.scatter(x, y, label="Training Data")
plt.plot(x, y_pred_line, label="Regression Line")
plt.scatter(new_x, prediction, label="Prediction", marker="x")
plt.legend()
plt.title("Linear Regression Prediction")
plt.show()

#plot loss over epochs

plt.plot(loss_history)
plt.title("Loss Over Training")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
