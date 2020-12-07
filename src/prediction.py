# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 11:56:49 2020
@author: Kiran Rawat
"""

# calculating predictions
def get_predictions(model,X_test):
    """
    get predicted labels
    Args: 
        model (sklearn.linear_model): statement column  
        X_test (pandas.core.frame.DataFrame): test data
    Returns:
        prediction
    """
    pred = model.predict(X_test)
    
    return pred