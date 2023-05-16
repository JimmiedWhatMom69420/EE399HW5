# EE 399 SPRING QUATER 2023
# Instructor: J. Nathan Kutz
# HOMEWORK #5:
# DUE: Midnight on 5/15 (Extra credit if turned in by 5/12)
# For the Lorenz equations (code given out previously in class emails), consider the following.

# github: https://github.com/JimmiedWhatMom69420/EE399HW5

# 1. Train a NN to advance the solution from t to t + ∆t for ρ = 10, 28 and 40. Now see how well
# your NN works for future state prediction for ρ = 17 and ρ = 35.

import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


from keras.models import Sequential
from keras.layers import Dense, LSTM, SimpleRNN, Flatten

import numpy as np
from tensorflow import keras

from reservoirpy.nodes import Reservoir, Ridge, Input
from sklearn.metrics import mean_squared_error

# Define the parameters for the Lorenz system
time_steps = 1000
step_size = 10
dt = 0.01
num_epochs = 100

# Generate Initial Lorenz system data
x0 = np.random.uniform(-20, 20)
y0 = np.random.uniform(-20, 20)
z0 = np.random.uniform(0, 50)
init = np.zeros((time_steps, 3))
init[0, :] = [x0, y0, z0]


def gen_lorenz(x, rhos):
    X = []
    Y = []

    for rho in rhos:
        for i in range(1, time_steps):
            dx_dt, dy_dt, dz_dt = lorenz(x[i - 1, 0], x[i - 1, 1], x[i - 1, 2], rho=rho)
            x[i, 0] = x[i - 1, 0] + dx_dt * dt
            x[i, 1] = x[i - 1, 1] + dy_dt * dt
            x[i, 2] = x[i - 1, 2] + dz_dt * dt

        # Create the input and output data for the model
        X_rho = np.zeros((time_steps - 10, 10, 3))
        Y_rho = np.zeros((time_steps - 10, 3))
        for i in range(10, time_steps):
            X_rho[i - 10, :, :] = x[i - 10:i, :]
            Y_rho[i - 10, :] = x[i, :]

        # Append the data for this rho value to the overall dataset
        X.append(X_rho)
        Y.append(Y_rho)

    # Combine the data for different rho values
    X = np.concatenate(X)
    Y = np.concatenate(Y)
    return X, Y


# Function to plot on rhos

# Define the Lorenz system equations
def lorenz(x, y, z, sigma=10, rho=28, beta=8/3):
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return dx_dt, dy_dt, dz_dt
def predict_on(x, rhos, net):
    figsize = (10, 5)
    if (len(rhos) == 3):
        figsize = (20, 5)

    # Create the figure for plotting the results
    fig, axs = plt.subplots(1, len(rhos), figsize=figsize, subplot_kw={'projection': '3d'})

    # Loop over different values of rho
    for j in range(len(rhos)):
        rho = rhos[j]
        for i in range(1, time_steps):
            dx_dt, dy_dt, dz_dt = lorenz(x[i - 1, 0], x[i - 1, 1], x[i - 1, 2], rho=rho)
            x[i, 0] = x[i - 1, 0] + dx_dt * dt
            x[i, 1] = x[i - 1, 1] + dy_dt * dt
            x[i, 2] = x[i - 1, 2] + dz_dt * dt

        # Create the input and output data for the model
        X = np.zeros((time_steps - 10, 10, 3))
        Y = np.zeros((time_steps - 10, 3))
        for i in range(10, time_steps):
            X[i - 10, :, :] = x[i - 10:i, :]
            Y[i - 10, :] = x[i, :]

        # Make predictions on the testing set
        y_pred = net.predict(X, verbose=0)

        # Evaluate the model on the testing set
        mse = net.evaluate(X, Y, verbose=0)
        print("Mean squared error:", mse, "for rho = {}".format(rho))

        # Plot the actual and predicted trajectories
        axs[j].plot(Y[:, 0], Y[:, 1], Y[:, 2], label='Actual')
        axs[j].plot(y_pred[:, 0], y_pred[:, 1], y_pred[:, 2], label='Predicted')
        axs[j].set_xlabel('X')
        axs[j].set_ylabel('Y')
        axs[j].set_zlabel('Z')
        axs[j].legend()
        axs[j].set_title('rho = {}'.format(rho))

    plt.show()

# Define the FNN model
net_ff = Sequential()
net_ff.add(Flatten(input_shape=(10, 3)))
net_ff.add(Dense(32, activation='sigmoid'))
net_ff.add(Dense(32, activation='sigmoid'))
net_ff.add(Dense(3, activation=None))

net_ff.compile(optimizer='adam', loss='mse')

# Compile the model
net_ff.compile(loss='mse', optimizer='adam')

X, Y = gen_lorenz(init, [10, 28, 40])

# Split the data into training and testing sets
train_size = int(0.9 * len(X))
train_X, test_X = X[:train_size,:,:], X[train_size:,:,:]
train_Y, test_Y = Y[:train_size,:], Y[train_size:,:]

# Train the model
net_ff.fit(train_X, train_Y, epochs=num_epochs, batch_size=32, verbose=2)

# Evaluate the model on the testing set
mse = net_ff.evaluate(test_X, test_Y, verbose=0)
print("Mean squared error:", mse)

# Make predictions on the testing set
y_pred = net_ff.predict(test_X, verbose=0)

predict_on(init, [10,28,40], net_ff)

predict_on(init, [17, 35], net_ff)