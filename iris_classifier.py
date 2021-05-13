####################################################################################################################
#
# File:        iris_classifier.py
# Description: SVM classifier model for iris dataset. Read the data from a table in postgresql database,
#              and then apply SVM classifier. At the end save the model for later usage.
# Author:      Mostafa Mamdouh
# Created:     Mon May 10 20:23:43 PDT 2021
#
####################################################################################################################


import numpy as np
import pandas as pd
import sqlalchemy as db


def main():
    '''
    # import iris dataset
    from sklearn import datasets
    df = datasets.load_iris()
    X = df.data 
    y = df.target
    '''
    
    # create connection with the database
    con = db.create_engine('postgresql://db:db@localhost/database1')
    
    # Create a SQL query to load the entire iris table
    query = """
    SELECT *
    FROM iris
    """
    
    # Load to dataframe
    df = pd.read_sql(query, con)
    df_desc = df.describe()
    
    # It is known that iris dataset has no missing data, so we proceed
    # Separate Dependnt and Independent Variable
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    # Encoding the Dependent Variable
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    
    # Training the Kernel SVM model on the Training set
    from sklearn.svm import SVC
    classifier = SVC(C=1, kernel='linear', random_state=0)
    classifier.fit(X_train, y_train)
    
    # provide some metrics
    from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy: {:.2f} %".format(acc*100))
    
    # Applying K-Fold Cross Validation after shuffling 
    # This usefule because sometimes data is sorted according to the output class [0,0,0, ... ,1,1,1, ..., 2,2,2,...]
    from sklearn.utils import shuffle
    from sklearn.model_selection import cross_val_score
    a_train, b_train = shuffle(X_train, y_train, random_state=0)
    accuracies = cross_val_score(estimator = classifier, X = a_train, y = b_train, cv = 4)
    print("Accuracy after K-Fold Cross Validation: {:.2f} %".format(accuracies.mean()*100))
    print("Standard Deviation after K-Fold Cross Validation: {:.2f} %".format(accuracies.std()*100))
    
    # Applying Grid-Search after shuffling 
    # This usefule because sometimes data is sorted according to the output class [0,0,0, ... ,1,1,1, ..., 2,2,2,...]
    from sklearn.model_selection import GridSearchCV
    a_train, b_train = shuffle(X_train, y_train, random_state=0)
    parameters = [{'C': [0.1, 0.25, 0.5, 0.75, 1], 'kernel': ['linear']},
                  {'C': [0.1, 0.25, 0.5, 0.75, 1], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
    grid_search = GridSearchCV(estimator = classifier,
                               param_grid = parameters,
                               scoring = 'accuracy',
                               cv = 4,
                               n_jobs = -1)
    grid_search.fit(a_train, b_train)
    best_accuracy = grid_search.best_score_
    best_parameters = grid_search.best_params_
    print("Best Accuracy after Grid Search: {:.2f} %".format(best_accuracy*100))
    print("Best Parameters after Grid Search:", best_parameters)
    
    '''
    # Save the model 
    import pickle
    with open(path, 'wb') as file:
        pickle.dump(classifier, file)
    '''
       
    # Save the model 
    from joblib import dump
    path = 'iris_model.joblib'
    dump(classifier, path) 
    

if __name__ == "__main__":
    main()
