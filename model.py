from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.linear_model import LinearRegression


# Create baseline model
def get_baseline_model():
    """
    Creates a simple Linear Regression model.
    This model acts as a baseline so we can compare its performance with the LSTM model.
    """
    # Create Linear Regression object
    baseline_model = LinearRegression()
    # Return the model
    return baseline_model


# Create LSTM model
def get_lstm_model(input_shape):
    """
    Creates an LSTM neural network for time series prediction.
    Parameters:
    input_shape: tuple
    Example:
    (6, 20)

    Meaning:
    - 6 time steps (lookback window)
    - 20 features
    """
    model = Sequential(
        [
            # LSTM Layer
            LSTM(
                units = 32, # Number of neurons
                activation = "tanh", # Activation function
                input_shape = input_shape, # Input shape
                return_sequences = False # Return final output only
            ),
            # Dropout Layer
            Dropout(0.2), # Drop 20% of neurons randomly
            # Output Layer
            Dense(units = 1) # Predict one value
        ]
    )
    # Compile Model
    model.compile(
        optimizer = "adam", # Learning algorithm
        loss = "mse" # Mean Squared Error
    )

    # Return completed model
    return model

