#NN
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from validations import discrete_evaluations, continuous_evaluations, tune_prob_threshold

# Check if MPS is available
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


class NNDataFromPd(Dataset):

    def __init__(self,X,y,data_dict):
        '''
        df: pandas dataframe
        df_dict: data dictionary of df
        '''
        ix_cols = [col + "_ix" for col in data_dict[(data_dict.encoded ==1)]['feature'].values]
        print(f"number of rows with Nulls: {X.isna().sum(axis = 1)}")

        X = X.fillna(-999)

        self.X_numeric_tensor = torch.tensor(X.drop(columns = ix_cols).values,dtype = torch.float32)
        self.X_categoric_tensor = torch.tensor(X.loc[:,ix_cols].values,dtype = torch.float32)
        self.y_tensor = torch.tensor(y.values,dtype= torch.float32)


    def __len__(self):
        return len(self.X_numeric_tensor)


    def __getitem__(self,ix):
        return self.X_numeric_tensor[ix], self.X_categoric_tensor[ix], self.y_tensor[ix]
        


class LogisticNNModel(nn.Module):

    def __init__(self,n_features):
        super().__init__()

        # for non normalized data:
        self.bn = nn.BatchNorm1d(n_features)
        self.linear1 = nn.Linear(n_features,1)
        self.sigmoid = nn.Sigmoid()
        

    def forward(self,x):
        x = self.bn(x)
        x = self.linear1(x)
        x = self.sigmoid(x)
        return x
    
class LogisticNNModelComplex(nn.Module):

    def __init__(self,n_features):
        super().__init__()


        self.model = nn.Sequential(
            
            nn.Linear(n_features,128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(.3),

            nn.Linear(128,64),
            nn.ReLU(),
            nn.Dropout(.2),

            nn.Linear(64,32),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Linear(32,1),
            
        )

    
    def forward(self,x):
        return self.model(x)

    
class MultiClassModel(nn.Module):
    

    def __init__(self, n_features,n_classes):
        super().__init__()

        self.bn = nn.BatchNorm1d(n_features)
        self.linear1 = nn.Linear(n_features,n_classes)
        
    
    def forward(self,x):
        x = self.bn(x)
        x = self.linear1(x)
        
        return x

class TorchClassifier(nn.Module):
    def __init__(self, n_features, n_hidden1=128, n_hidden2=64, n_hidden3=32, dropout_rate=0.3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(n_features, n_hidden1),
            nn.BatchNorm1d(n_hidden1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(n_hidden1, n_hidden2),
            nn.ReLU(),
            nn.Dropout(dropout_rate - 0.1),

            nn.Linear(n_hidden2, n_hidden3),
            nn.BatchNorm1d(n_hidden3),
            nn.ReLU(),

            nn.Linear(n_hidden3, 1)
        )

    def forward(self, x):
        return self.model(x)

def dl_classifier(train_X, test_X, train_y, test_y, dat_dict, output_path, problem_type=None, epochs=20, batch_size=64, learning_rate=0.001):
    """
    Trains a PyTorch neural network for classification or regression, similar to
    the scikit-learn pipelines in models.py.
    """
    # Create Copies to work within the function
    train_X = train_X.copy()
    test_X = test_X.copy()

    # Infer problem type if not specified
    if problem_type is None:
        if pd.api.types.is_float_dtype(train_y) or (pd.Series(train_y).nunique() > 20 and pd.api.types.is_numeric_dtype(train_y)):
            problem_type = 'regression'
        else:
            problem_type = 'classification'
    print(f"Inferred problem type: {problem_type}")

    categorical_features = dat_dict[(dat_dict['type'] == 'categorical') & (dat_dict['modeling_feature'] == 1)]['feature'].values
    numerical_features = [col for col in train_X.columns if col not in categorical_features]

    # Preprocessing
    # Ordinal Encoding for categorical features
    ordinal_encodings = {}
    for col in categorical_features:
        if col in train_X.columns:
            print(f"encoding feature {col}")
            oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            train_X[col] = oe.fit_transform(train_X[[col]].fillna('missing'))
            test_X[col] = oe.transform(test_X[[col]].fillna('missing'))
            ordinal_encodings[col] = oe
            joblib.dump(oe, os.path.join(output_path, f'encodings/{col}_categorical_encodings.pkl'))

    # Scaling numerical features
    if numerical_features:
        scaler = StandardScaler()
        train_X[numerical_features] = scaler.fit_transform(train_X[numerical_features].fillna(0))
        test_X[numerical_features] = scaler.transform(test_X[numerical_features].fillna(0))
        joblib.dump(scaler, os.path.join(output_path, 'scaler.pkl'))
    
    # Fill any remaining NaNs - safety net
    train_X.fillna(-999, inplace=True)
    test_X.fillna(-999, inplace=True)

    # Convert to tensors
    X_train_tensor = torch.tensor(train_X.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(train_y.values, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(test_X.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(test_y.values, dtype=torch.float32).view(-1, 1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model, Loss, Optimizer
    n_features = train_X.shape[1]
    model = TorchClassifier(n_features).to(device)

    if problem_type == 'classification':
        criterion = nn.BCEWithLogitsLoss()
    else: # regression
        criterion = nn.MSELoss()
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader):.4f}')

    # Evaluation
    model.eval()
    with torch.no_grad():
        if problem_type == 'classification':
            # Training Metrics
            train_outputs = model(X_train_tensor.to(device))
            train_pred_proba = torch.sigmoid(train_outputs).cpu().numpy().flatten()
            train_pred = (train_pred_proba > 0.5).astype(int)
            print("Training Metrics: \n")
            discrete_evaluations(train_y, train_pred, train_pred_proba, 'train', classification_type="binomial", model_path=output_path)

            # Test Metrics
            test_outputs = model(X_test_tensor.to(device))
            pred_proba = torch.sigmoid(test_outputs).cpu().numpy().flatten()
            pred = (pred_proba > 0.5).astype(int)
            print("Test Metrics: \n")
            discrete_evaluations(test_y, pred, pred_proba, 'test', classification_type="binomial", model_path=output_path)

            # Threshold tuning
            thres_new = tune_prob_threshold(test_y, pred_proba)
            pred_new = np.where(pred_proba > thres_new['tpr_fpr'] - 0.03, 1, 0)
            thres_df = pd.DataFrame(list(thres_new.items()), columns=['threshold', 'value'])
            thres_df.to_csv(os.path.join(output_path, "thresholds.csv"), index=False)
            print("\n Evaluations after probability threshold tuning: ")
            discrete_evaluations(test_y, pred_new, pred_proba, 'test_parameter_tuned', classification_type="binomial", model_path=output_path)

        else: # regression
            # Train metrics
            train_pred = model(X_train_tensor.to(device)).cpu().numpy().flatten()
            print("Training Metrics (Regression): \n")
            continuous_evaluations(train_y, train_pred, 'train')

            # Test metrics
            pred = model(X_test_tensor.to(device)).cpu().numpy().flatten()
            print("Test Metrics (Regression): \n")
            continuous_evaluations(test_y, pred, 'test')

    # Save model
    torch.save(model.state_dict(), os.path.join(output_path, 'dl_classifier.pth'))
    dat_dict.to_csv(os.path.join(output_path, 'dat_dict.csv'), index=False)

    return model

    
