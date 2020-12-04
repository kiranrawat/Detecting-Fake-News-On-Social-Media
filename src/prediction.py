# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 11:56:49 2020
@author: Kiran Rawat
"""

# calculating predictions
def get_predictions(model,X_test):
    """get predicted labels
    """
    pred = model.predict(X_test)
    
    return pred