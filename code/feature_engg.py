
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

"""
This is a code to implement feature engineering on the dataset
"""
class feature_selection_and_engg:


    def feature_engineering(self,df):
        """
        Peform feature engineering steps:
        1. Fixing the money and percents#
        2. Creating the train/val/test set
        3. smashing sets back together
        Args:
        -   dataframe
        Returns:
        -   dataframe
        """
        # 1. Fixing the money and percents#
        df['x12'] = df['x12'].str.replace('$', '')
        df['x12'] = df['x12'].str.replace(',', '')
        df['x12'] = df['x12'].str.replace(')', '')
        df['x12'] = df['x12'].str.replace('(', '-')
        df['x12'] = df['x12'].astype(float)
        df['x63'] = df['x63'].str.replace('%', '')
        df['x63'] = df['x63'].astype(float)

        return df

    def feature_selection(self,df):
        """
        Looking at the correlation map from above, we can see there are very few variables associated with the dependent variable.
        Thus, we will use an L1 penalty to for feature selection. Interestingly enough, we see a that some variables have heavy
        correlation amongst themselves.
        Args:
        -   dataframe
        Returns:
        -   list
        """
        exploratory_LR = LogisticRegression(penalty='l1', fit_intercept=False, solver='liblinear')
        exploratory_LR.fit(df.drop(columns=['y']), df['y'])
        exploratory_results = pd.DataFrame(df.drop(columns=['y']).columns).rename(columns={0: 'name'})
        exploratory_results['coefs'] = exploratory_LR.coef_[0]
        exploratory_results['coefs_squared'] = exploratory_results['coefs'] ** 2
        var_reduced = exploratory_results.nlargest(25, 'coefs_squared')
        variables = var_reduced['name'].to_list()

        return variables

    def convert_bool_to_int_dtype(self,df):
        """
        This function will fix boolean datatype to int datatype
        Args:
        -   dataframe
        Returns:
        -   dataframe
        """
        g = df.columns.to_series().groupby(df.dtypes).groups

        list_of_column = []
        for k, v in g.items():
            #print(k.name)
            if k.name == "bool":
                #print("Inside")
                list_of_column.append(v)


        for i in range(len(list_of_column)):
            df[list_of_column[i]] = df[list_of_column[i]].astype(int)

        return df

    def df_imputation(self,df):
        """
        Implementing mean imputation on train dataset
        Args:
        -   dataframe
        Returns:
        -   dataframe
        """

        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        df_imputed = pd.DataFrame(imputer.fit_transform(df.drop(columns=['y', 'x5', 'x31', 'x81', 'x82'])),
                                     columns=df.drop(columns=['y', 'x5', 'x31', 'x81', 'x82']).columns)
        std_scaler = StandardScaler()
        df_imputed_std = pd.DataFrame(std_scaler.fit_transform(df_imputed), columns=df_imputed.columns)

        # 3 create dummies

        dumb5 = pd.get_dummies(df['x5'], drop_first=True, prefix='x5', prefix_sep='_', dummy_na=True)
        df_imputed_std = pd.concat([df_imputed_std, dumb5], axis=1, sort=False)

        dumb31 = pd.get_dummies(df['x31'], drop_first=True, prefix='x31', prefix_sep='_', dummy_na=True)
        df_imputed_std = pd.concat([df_imputed_std, dumb31], axis=1, sort=False)

        dumb81 = pd.get_dummies(df['x81'], drop_first=True, prefix='x81', prefix_sep='_', dummy_na=True)
        df_imputed_std = pd.concat([df_imputed_std, dumb81], axis=1, sort=False)

        dumb82 = pd.get_dummies(df['x82'], drop_first=True, prefix='x82', prefix_sep='_', dummy_na=True)
        df_imputed_std = pd.concat([df_imputed_std, dumb82], axis=1, sort=False)
        df_imputed_std = pd.concat([df_imputed_std, df['y']], axis=1, sort=False)

        del dumb5, dumb31, dumb81, dumb82

        return df_imputed_std

