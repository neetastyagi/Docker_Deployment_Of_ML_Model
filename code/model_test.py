import pandas as pd
import statsmodels.api as sm
import pickle
import feature_engg as feature_selection_and_engg


"""
This is a code to implement GLM Model and then saving the model 
"""
class GLM_Model:
    """

    """

    def triangle(x: int, y: int, z: int) -> str:
        if x == y == z:
            return "Equilateral triangle"
        if x in {y, z} or y == z:
            return "Isosceles triangle"
        return "Scalene triangle"
    # def load_dataset(self, path):
    #     """
    #     This will be used to load the dataset for GLM model
    #
    #     Args:
    #     -   path
    #     Returns:
    #     -   dataframe
    #     """
    #     df = pd.read_csv(path)
    #
    #     return df

    # def model(self,y,X):
    #     """
    #     This function will build model
    #     Args:
    #     -   dataframe
    #     Returns:
    #     -   object
    #     """
    #     logit = sm.Logit(y, X)
    #
    #     return logit
    #
    # def model_train(self, model):
    #     """
    #     This function will build model
    #     Args:
    #     -   onject
    #     Returns:
    #     -   object
    #     """
    #
    #     result = model.fit()
    #
    #     return result
    #
    #
    #
    # def predict(self, model, X):
    #     """
    #     This function will predict the outcome of the given dataset
    #     Args:
    #     -   object, dataframe
    #     Returns:
    #     -   dataframe
    #     """
    #     outcome = pd.DataFrame(model.predict(X)).rename(columns={0: 'probs'})
    #
    #
    #     return outcome
    #
    # def save_pickle_file(self, model):
    #     """
    #     This function saves the model as pickle fil
    #     Args:
    #     -   object
    #     Returns:
    #     -   None
    #     """
    #
    #     with open('GLM.pkl', 'wb') as model_pkl:
    #         pickle.dump(model, model_pkl, protocol=2)
    #

if __name__=='__main__':

    # creating the object for the class GLM_model
    GLM_Model = GLM_Model()
    # Loading train data
    GLM_Model.triangle(2,3, 4)
    #raw_train = GLM_Model.load_dataset('exercise_26_train.csv')
    # # creating the object for the class feature_selection_and_engg
    # feature_selection_and_engg = feature_selection_and_engg.feature_selection_and_engg()
    # # Perform feature engineering on the train data
    # raw_train = feature_selection_and_engg.feature_engineering(raw_train)
    # # Perform data imputation
    # all_train_imputed_std = feature_selection_and_engg.df_imputation(raw_train)
    # # select variables
    # variables = feature_selection_and_engg.feature_selection(all_train_imputed_std)
    # # select features for training data
    # all_train_X = feature_selection_and_engg.convert_bool_to_int_dtype(all_train_imputed_std[variables])
    # # passing train data to the model
    # model = GLM_Model.model(all_train_imputed_std['y'], all_train_X)
    # # Fit the model
    # result = GLM_Model.model_train(model)
    # # Save the model as pickle file
    # GLM_Model.save_pickle_file(result)
