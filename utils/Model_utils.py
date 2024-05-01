import numpy as np
from itertools import product
from sklearn.metrics import fbeta_score, recall_score, precision_score

class GridThreshSearch:
    def __init__(self, model, params, static_params):
        self.model = model
        self.params = params
        self.static_params = static_params


    def fit(self, x_train, y_train, x_val, y_val):
        self.best_rec = 0
        self.best_prec = 0        
        self.best_params = 0
        self.best_th = 0
        self.best_fb = 0
        self.best_estimator = ''
        for par in self.params:
            par.update(self.static_params)
            mod = self.model(**par)
            mod.fit(x_train, y_train) 
            pred_probs = mod.predict_proba(x_val)[:, 1]
            for prob in np.arange(0, 1, 0.1):
                y_hat = (pred_probs > prob).astype(int)
                rec = recall_score(y_val, y_hat)
                prec = precision_score(y_val, y_hat)                
                fb = fbeta_score(y_val, y_hat, beta=2)                
                if fb > self.best_fb:
                    self.best_rec = rec
                    self.best_prec = prec                
                    self.best_fb = fb
                    self.best_params = par
                    self.best_th = prob
                    self.best_estimator = mod
        return self



def param_maker(parameters:dict):
    
    keys = list(parameters.keys())
    vals = list(parameters.values())

    vals = list(product(*vals))

    result = ({keys[i]: val[i] for i in range(len(keys))} for val in vals)

    return result
