#NN
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset


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

            nn.Linear(32,1)
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


    
