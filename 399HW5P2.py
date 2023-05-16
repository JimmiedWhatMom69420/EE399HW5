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

# Define the Lorenz system equations
def lorenz(x, y, z, sigma=10, rho=28, beta=8/3):
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return dx_dt, dy_dt, dz_dt

# Create the ESN model (YOU WILL NEED RESERVIORPY)
data = Input(input_dim=3)
reservoir_node = Reservoir(units=100, size=100, leak_rate=.3, spectral_radius=.9)
readout_node = Ridge(output_dim=3, ridge=1e-2)

net_esn = data >> reservoir_node >> readout_node

rhos = [10, 28, 40]

for j in range(len(rhos)):
    rho = rhos[j]

    # Generate the training data
    np.random.seed(42)
    t = np.arange(0, time_steps, dt)
    x = np.zeros_like(t)
    y = np.zeros_like(t)
    z = np.zeros_like(t)
    x[0], y[0], z[0] = x0, y0, z0
    [x[0], y[0], z[0]] = init[0, :]

    for i in range(1, time_steps):
        x_dot, y_dot, z_dot = lorenz(x[i - 1], y[i - 1], z[i - 1], rho=rho)
        x[i] = x[i - 1] + x_dot * dt
        y[i] = y[i - 1] + y_dot * dt
        z[i] = z[i - 1] + z_dot * dt

    # Split the data into input/output pairs
    input_data = np.column_stack([x, y, z])
    output_data = np.zeros_like(input_data)
    for i in range(len(t)):
        x_dot, y_dot, z_dot = lorenz(x[i], y[i], z[i], rho=rho)
        output_data[i, :] = [x_dot, y_dot, z_dot]

    # Split the data into training and testing sets
    train_size = int(0.9 * len(input_data))
    train_X, test_X = input_data[:train_size, :], input_data[train_size:, :]
    train_Y, test_Y = output_data[:train_size, :], output_data[train_size:, :]

    # Train the ESN
    net_esn.fit(train_X, train_Y)

    # Evaluate the ESN
    y_pred = net_esn.run(test_X)

    mse = mean_squared_error(test_Y, y_pred)
    print(f"Mean squared error: {mse}", "for rho = {}".format(rho))

    rhos = [17, 35]

    # Create the figure for plotting the results
    fig, axs = plt.subplots(1, len(rhos), figsize=(10, 5), subplot_kw={'projection': '3d'})

    for j in range(len(rhos)):
        rho = rhos[j]
        [x[0], y[0], z[0]] = init[0, :]

        for i in range(1, len(t)):
            x_dot, y_dot, z_dot = lorenz(x[i - 1], y[i - 1], z[i - 1], rho=rho)
            x[i] = x[i - 1] + x_dot * dt
            y[i] = y[i - 1] + y_dot * dt
            z[i] = z[i - 1] + z_dot * dt

        # Split the data into input/output pairs
        input_data = np.column_stack([x, y, z])
        output_data = np.zeros_like(input_data)
        for i in range(len(t)):
            x_dot, y_dot, z_dot = lorenz(x[i], y[i], z[i])
            output_data[i, :] = [x_dot, y_dot, z_dot]

        # Evaluate the ESN
        y_pred = net_esn.run(input_data)
        mse = mean_squared_error(output_data, y_pred)
        print(f"Mean squared error: {mse}")

        # Plot the actual and predicted trajectories
        axs[j].plot(output_data[:, 0], output_data[:, 1], output_data[:, 2], label='Actual')
        axs[j].plot(y_pred[:, 0], y_pred[:, 1], y_pred[:, 2], label='Predicted')
        axs[j].set_xlabel('X')
        axs[j].set_ylabel('Y')
        axs[j].set_zlabel('Z')
        axs[j].set_title(f'Lorenz System with Rho = {rho}')
        axs[j].legend()
    plt.show()