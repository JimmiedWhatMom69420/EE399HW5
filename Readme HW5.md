# EE399 HW-5

``Author: Marcel Ramirez``

``Publish Date: 5/15/2023``

``Course: Spring 2023 EE399``

##### This assignment focuses on training neural networks to predict what may happen in the future utilizing Lorenz equations for the Lorenz system.

# Summary

This assignment focuses on the performance of several types of neural networks in a Lorenz system. Below are two key factors involved:

1.  Train a Neural Network (NN): Create a NN model that takes the current state at time t as input and predicts the next state at time t + ∆t for different values of the parameter ρ (10, 28, and 40). Train the NN using the provided Lorenz system data. Evaluate the performance of the trained NN by predicting future states for ρ = 17 and ρ = 35.
2.  Compare Different Neural Network Architectures: Implement and compare different types of neural network architectures for forecasting the dynamics of the Lorenz system. Specifically, compare feed-forward networks, Long Short-Term Memory (LSTM) networks, Recurrent Neural Networks (RNNs), and Echo State Networks (ESNs). Use the same Lorenz system data to train and evaluate these architectures. Assess and compare their performance in terms of accuracy and predictive capability.

## Theory
The lorenz equations describe the qualities of a simpler version of a real physical system. Being able to describe one by including a popular benchmark to study chaotic systems. The lorenz system was named after Edward Lorenz in 1963 who discovered this. 

Below is a rundown of the differential equations to describe atmospheric convection in a simple model of dynamics:

$$dy/dt = x(ρ - z) - y$$
$$dx/dt = σ(y - x)$$
$$dz/dt = xy - βz $$

1.  State Prediction with Neural Networks: The task of training a Neural Network (NN) to advance the solution from time t to t + ∆t involves approximating the underlying dynamics of the Lorenz system. Given the current state at time t, the NN is trained to predict the next state at time t + ∆t. By adjusting the weights and biases of the NN through a training process, it learns to capture the nonlinear relationships between the system variables and make accurate predictions.
    
2.  Comparison of Neural Network Architectures: In the second task, different types of neural network architectures are compared for forecasting the dynamics of the Lorenz system. These architectures include feed-forward networks, Long Short-Term Memory (LSTM) networks, Recurrent Neural Networks (RNNs), and Echo State Networks (ESNs).
    

-   Feed-forward networks: These networks consist of layers of interconnected neurons, where information flows in one direction, from the input layer to the output layer. They are commonly used for tasks where the current input alone is sufficient to make predictions.
    
-   LSTM networks: LSTM networks are a type of recurrent neural network that can model long-term dependencies in sequential data. They incorporate memory cells that allow them to capture and remember information from previous time steps. This makes LSTM networks well-suited for time series forecasting tasks.
    
-   RNNs: Recurrent Neural Networks are another type of neural network architecture that can process sequential data. They maintain an internal hidden state that is updated at each time step, enabling them to capture temporal dependencies. However, traditional RNNs suffer from the "vanishing gradient" problem, limiting their ability to capture long-term dependencies.
    
-   Echo State Networks: ESNs are a type of recurrent neural network with a fixed random structure. The weights of the recurrent connections are randomly initialized and remain unchanged during training, while the output layer is trained using a linear regression algorithm. ESNs have been shown to be effective in capturing the dynamics of complex systems.
    

By comparing these different architectures, the assignment aims to evaluate their performance in terms of their ability to accurately forecast the dynamics of the Lorenz system. This assessment can provide insights into which architecture is most suitable for modeling and predicting chaotic systems.

# Code walkthrough

### Question 1
Before approaching question 1 and especially question 2, we must first develop the Lorenz System. 

```python
def lorenz_deriv(x_y_z, t0, sigma=10, beta=8/3, rho=rho):
    x, y, z = x_y_z
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

```
We also want to store the Lorenz system trajectories and plot them just as well. We first want to establish what the variables are. These would be dt, T, t, beta, sigma, and rho.

```python
fig, ax = plt.subplots(1, 1, subplot_kw={'projection': '3d'})
for j in range(100):
    nn_input[j * (len(t) - 1):(j + 1) * (len(t) - 1), :] = x_t[j, :-1, :]
    nn_output[j * (len(t) - 1):(j + 1) * (len(t) - 1), :] = x_t[j, 1:, :]
    x, y, z = x_t[j, :, :].T
    ax.plot(x, y, z, linewidth=1)
    ax.scatter(x0[j, 0], x0[j, 1], x0[j, 2], color='r')
ax.view_init(18, -113)
plt.show()

```

To train a neural network (NN) to advance the solution from time t to t + ∆t for different values of ρ and evaluate its performance for future state prediction, you need to make the following modifications to the given code:

1.  Update the value of ρ to the desired values (e.g., 10, 28, and 40) by modifying the `rho` variable in the code.
    
2.  Modify the code to split the dataset into training and testing sets. Currently, the entire dataset is used for training. You can use the `train_test_split` function from scikit-learn to split the dataset into training and testing sets.
    
3.  Update the model architecture and training parameters accordingly.

Now we train the script on a neural network to advance the solution from ``t to t + Δt``  for ρ = 10, 28, 40

![image](https://media.discordapp.net/attachments/823976012203163649/1108644652447449088/FNq9eABJNdAAAAAElFTkSuQmCC.png?width=1438&height=443)
Next, lets see the computational output for ρ = 17 and 35 

![5eGgV0GDP00AAAAASUVORK5CYII.png (916×426) (discordapp.net)](https://media.discordapp.net/attachments/823976012203163649/1108644651377905684/5eGgV0GDP00AAAAASUVORK5CYII.png?width=1296&height=603)


### Question 2

To compare feed-forward, LSTM, RNN and Echo State Netowrks for forecasting the dynamics of the Lorenz system, we must briefly follow these:

1.  The Lorenz equations and system parameters are defined.
    
2.  Initial conditions are generated for the Lorenz system.
    
3.  Input and output arrays are prepared for the neural networks by integrating the Lorenz equations.
    
4.  The data is split into training and testing sets.
    
5.  Feed-Forward Neural Network (FNN):
    
    -   A feed-forward neural network model is created using the Keras Sequential API.
    -   The model is trained on the training data using the mean squared error loss.
    -   The model makes predictions on the testing data.
6.  LSTM:
    
    -   An LSTM model is created using the Keras Sequential API.
    -   The model is trained on the training data using the mean squared error loss.
    -   The model makes predictions on the testing data.
7.  RNN:
    
    -   An RNN model is created using the Keras Sequential API.
    -   The model is trained on the training data using the mean squared error loss.
    -   The model makes predictions on the testing data.
8.  Echo State Network (ESN):
    
    -   An Echo State Network model is created using the `reservoirpy` library.
    -   The model is trained on the training data.
    -   The model makes predictions on the testing data.
9.  The predictions from each model are compared using mean squared error (MSE) as the evaluation metric.
    

The code allows for a comparison of the performance of different neural network models in forecasting the dynamics of the Lorenz system.

Below is the computational output for it. Running all models giving the mean square errors for rho of 10 and 35

---------------------
| Mean Squared Error | Rho |
| -- | -- |
| 0.15224202827965347 | 10 |
| 2331.242977221156 | |
| 10145.713553004734 ||
|1.7928471466364329 | 35 |
| 2290.874895267006 ||
|9785.374156340838 ||

Below is the code for the models in python:

```python
# Feed-Forward Neural Network (FNN) 
model_ff = Sequential() 
model_ff.add(Dense(32, activation='relu', input_shape=(3,))) model_ff.add(Dense(32, activation='relu')) model_ff.add(Dense(3)) model_ff.compile(optimizer='adam', loss='mse') model_ff.fit(X_train, Y_train, epochs=50, batch_size=32, verbose=0) Y_pred_ff = model_ff.predict(X_test) 

# LSTM 
model_lstm = Sequential() model_lstm.add(LSTM(32, activation='relu', input_shape=(1, 3))) 
model_lstm.add(Dense(3)) 
model_lstm.compile(optimizer='adam', loss='mse') model_lstm.fit(X_train[:, np.newaxis, :], Y_train, epochs=50, batch_size=32, verbose=0) Y_pred_lstm = model_lstm.predict(X_test[:, np.newaxis, :]) 
# RNN 
model_rnn = Sequential() 
model_rnn.add(SimpleRNN(32, activation='relu', input_shape=(1, 3))) model_rnn.add(Dense(3)) model_rnn.compile(optimizer='adam', loss='mse') 
model_rnn.fit(X_train[:, np.newaxis, :], Y_train, epochs=50, batch_size=32, verbose=0) Y_pred_rnn = model_rnn.predict(X_test[:, np.newaxis, :]) 
# Echo State Network (ESN) 
model_esn = ESN(n_inputs=3, n_outputs=3, n_reservoir=1000) model_esn.fit(X_train, Y_train) Y_pred_esn = model_esn.predict(X_test)
```

Below is the computational output for this code: 

```python
Running Model-1:   0%|          | 0/1 [00:00<?, ?it/s]

Running Model-1:   0%|          | 0/1 [00:00<?, ?it/s]

Running Model-1: 732it [00:00, 7301.54it/s]          

…

Running Model-1: 90000it [00:12, 7178.04it/s]

Fitting node Ridge-0...

Running Model-1: 100%|██████████| 1/1 [00:12<00:00, 12.58s/it]

Running Model-1: 10000it [00:01, 8259.51it/s]

Mean squared error: 0.15224202827965347 for rho = 10

Running Model-1: 100000it [00:11, 8343.80it/s]

Mean squared error: 2331.242977221156

Running Model-1: 100000it [00:12, 7821.03it/s]

Mean squared error: 10145.713553004734  
_____

Running Model-1:   0%|          | 0/1 [00:00<?, ?it/s]

Running Model-1:   0%|          | 0/1 [00:00<?, ?it/s]

Running Model-1: 713it [00:00, 7107.06it/s]          

…

Running Model-1: 90000it [00:12, 7310.28it/s]

Fitting node Ridge-0...

Running Model-1: 100%|██████████| 1/1 [00:12<00:00, 12.35s/it]

Running Model-1: 10000it [00:01, 8144.66it/s]

Mean squared error: 1.7928471466364329 for rho = 35

Running Model-1: 100000it [00:12, 8259.11it/s]

Mean squared error: 2290.874895267006

Running Model-1: 100000it [00:12, 7923.76it/s]

Mean squared error: 9785.374156340838
```
