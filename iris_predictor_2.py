####################################################################################################################
#
# File:        iris_predictor_2.py
# Description: SVM predictor model for iris dataset, it uses pickle to load the model.
# Author:      Mostafa Mamdouh
# Created:     Mon May 10 20:23:43 PDT 2021
#
####################################################################################################################

import pickle


def main():   
    # to predict
    X_test = [[3, 3, 3, 3]]
    
    # load the model
    path = r'iris_model.pkl'
    with open(path, 'rb') as file:
        classifier = pickle.load(file)
        
    # predict using the saved model
    y_pred_encoded = int(classifier.predict(X_test))
    print(y_pred_encoded)
    
    # Decoding the Dependent Variable
    y_pred = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}.get(y_pred_encoded, 'model error') 
    print(y_pred)


if __name__ == "__main__":
    main()

