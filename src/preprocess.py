import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(rentals_path, transaction_path):
    """
    Preprocess the rental and transactions data for the AVM model.
    
    Parameters:
    rentals_path (str): Path to the rental transactions CSV file.
    transaction_path (str): Path to the transactions transactions CSV file.
    
    Returns:
    pandas.DataFrame: Preprocessed and combined dataset.
    """
    # Load rental and transactions data
    rentals = pd.read_csv(rentals_path)
    transactions = pd.read_csv(transaction_path)
    
    # Combine rental and transactions data
    data = pd.concat([rentals, transactions], ignore_index=True)
    
    # Handle missing data
    imputer = SimpleImputer(strategy='median')
    data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    
    # Encode categorical variables
    categorical_cols = data.select_dtypes(include=['object']).columns
    encoder = LabelEncoder()
    data[categorical_cols] = data[categorical_cols].apply(encoder.fit_transform)
    
    # Scale numerical features
    scaler = StandardScaler()
    data[data.select_dtypes(include=['int64', 'float64']).columns] = scaler.fit_transform(data.select_dtypes(include=['int64', 'float64']))
    
    return data