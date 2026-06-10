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



