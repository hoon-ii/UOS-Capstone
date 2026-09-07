#%%
import pandas as pd
#%%
"""
Reference : https://archive.ics.uci.edu/dataset/186/wine+quality
"""
def load_raw_data(config):
    if config["dataset"] == "redwine":
        try:
            data = pd.read_csv('../data/winequality-red.csv', delimiter=";")
        except:
            data = pd.read_csv('./datasets/data/winequality-red.csv', delimiter=";")
        columns = list(data.columns)
        columns.remove("quality")
        
        assert data.isna().sum().sum() == 0
        
        continuous_features = columns
        categorical_features = [
            "quality"
        ]
        integer_features = []
        ClfTarget = "quality"
        
    elif config["dataset"] == "whitewine":
        data = pd.read_csv('../data/winequality-white.csv', delimiter=";")
        columns = list(data.columns)
        columns.remove("quality")
        
        assert data.isna().sum().sum() == 0
        
        continuous_features = columns
        categorical_features = [
            "quality"
        ]
        integer_features = []
        ClfTarget = "quality"
        
    return data, continuous_features, categorical_features, integer_features, ClfTarget

# %%