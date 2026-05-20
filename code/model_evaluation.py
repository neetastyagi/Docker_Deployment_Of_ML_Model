import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import pickle
import sys
import bokeh
import feature_engg as feature_selection_and_engg

"""
This is a code to implement GLM Model and then saving the model 
"""
class GLM_Model_Evaluation:
    """

    """
    def load_dataset(self, path):
        """
        This will be used to load the dataset for GLM model

        Args:
        -   path
        Returns:
        -   dataframe
        """
        df = pd.read_csv(path)

        return df

    def split_dataset(self,df):
        """
        Split the data into train, test and validation dataset
        Args:
        -   dataframe
        Returns:
        -   train, test, val dataframe
        """


        # 2. Creating the train/val/test set
        x_train, x_val, y_train, y_val = train_test_split(df.drop(columns=['y']), df['y'], test_size=0.1,
                                                          random_state=13)
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=4000, random_state=13)

        # 3. smashing sets back together
        train = pd.concat([x_train, y_train], axis=1, sort=False).reset_index(drop=True)
        val = pd.concat([x_val, y_val], axis=1, sort=False).reset_index(drop=True)
        test = pd.concat([x_test, y_test], axis=1, sort=False).reset_index(drop=True)

        return train, val, test

    def model(self,y,X):
        """
        This function will build model
        Args:
        -   dataframe
        Returns:
        -   model
        """
        logit = sm.Logit(y, X)

        return logit

    def model_train(self, model):
        """
        This function will build model
        Args:
        -   dataframe
        Returns:
        -   model
        """

        result = model.fit()
        #print(result.summary())
        with open('resultsummary.txt', 'w') as fh:
            fh.write(result.summary().as_text())

        return result



    def predict(self, model, X):
        """
        This function will predict the outcome of the given dataset
        Args:
        -   model, X
        Returns:
        -   dataframe
        """
        outcome = pd.DataFrame(model.predict(X)).rename(columns={0: 'probs'})


        return outcome



if __name__=='__main__':
    print("python version " + sys.version)
    print('numpy version ' + np.__version__)
    print('pandas version ' + pd.__version__)
    print('sklearn version ' + '0.23.1')
    print('bokeh version ' + bokeh.__version__)
    print('statsmodels version ' + '0.9.0')

    # creating the object for the class GLM_model
    GLM_Model_Evaluation = GLM_Model_Evaluation()
    # Loading train data
    raw_train = GLM_Model_Evaluation.load_dataset('exercise_26_train.csv')
    #creating the object for the class feature_selection_and_engg
    feature_selection_and_engg = feature_selection_and_engg.feature_selection_and_engg()
    raw_train = feature_selection_and_engg.feature_engineering(raw_train)

    train, val, test = GLM_Model_Evaluation.split_dataset(raw_train)

    train_imputed_std = feature_selection_and_engg.df_imputation(train)
    test_imputed_std = feature_selection_and_engg.df_imputation(test)
    val_imputed_std = feature_selection_and_engg.df_imputation(val)



    variables = feature_selection_and_engg.feature_selection(train_imputed_std)

    train_imputed_std_X = feature_selection_and_engg.convert_bool_to_int_dtype(train_imputed_std[variables])
    val_imputed_std_X = feature_selection_and_engg.convert_bool_to_int_dtype(val_imputed_std[variables])
    test_imputed_std_X = feature_selection_and_engg.convert_bool_to_int_dtype(test_imputed_std[variables])



    model = GLM_Model_Evaluation.model(train_imputed_std['y'], train_imputed_std_X)
    result = GLM_Model_Evaluation.model_train(model)

    #outcome = GLM_Model_Evaluation.predict(result,test_imputed_std_X)
    outcome = pd.DataFrame(GLM_Model_Evaluation.predict(result, test_imputed_std_X)).rename(columns={0: 'probs'})
    print(outcome)



