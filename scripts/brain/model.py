import torch
import torch.nn as nn

class RetailSalesPredictor(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=64):
        """
        A Light Deep Learning Model for Retail Sales Prediction.
        Uses Residual connections (ResNet-style) for 'Deep' learning capabilities on tabular data.
        """
        super(RetailSalesPredictor, self).__init__()
        
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        
        # Residual Block 1
        self.res_block1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )
        
        # Residual Block 2
        self.res_block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )
        
        # Output Head
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1) # Predict scalar Sales
        )
        
    def forward(self, x):
        # Input -> Hidden
        x = self.relu(self.bn1(self.input_layer(x)))
        
        # ResBlock 1
        identity = x
        out = self.res_block1(x)
        x = self.relu(out + identity) # Skip connection
        
        # ResBlock 2
        identity = x
        out = self.res_block2(x)
        x = self.relu(out + identity) # Skip connection
        
        # Output
        sales = self.output_head(x)
        return sales
