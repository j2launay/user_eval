from sklearn.preprocessing import LabelEncoder
import utils
import numpy as np
import pandas as pd

def generate_dataset(dataset_name):
    """Function to store the data and all the information about the dataset and features 

    Args:
        dataset_name (string): name of the dataset

    Returns:
        x_data: data
        y_data: labels of the data
        dataframe: pandas dataframe
        class_names: array of string corresponding to the names of the label
        continuous_features, categorical_features: array of int indicating the position
                                of the features with continuous or categorical features
        categorical_values: array of the possible values (integer) for each categorical feature
        categorical_names: dict of the original values (string) for each categorical feature
        feature_names: array of the name of the features 
        transformations, feature_transformations: functions used to modify the feature values 
        modify_feature_name: function to get back to the original value of categorical features
    """
    # Function used to get dataset depending on the dataset name
    class_names, transformations, dataframe = None, None, None
    categorical_values, categorical_names, modify_feature_name = [], {}, {}
    
    if "obesity" in dataset_name:
        dataset = pd.read_csv("./dataset/obesity.csv")
        try:
            dataset.drop(['Unnamed: 0'], axis=1, inplace=True)
        except KeyError:
            print()
        x_data = dataset.loc[:, dataset.columns != 'NObeyesdad']
        y_data = dataset.iloc[:,-1].values
        feature_names = dataset.columns[:-1]
        categorical_features = [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        tab = [i for i in range(len(x_data.iloc[0]))]
        categorical_values =[]

        feature_transformations = {}
        categorical_names = {}
        for feature in categorical_features:
            le = LabelEncoder()
            le.fit(x_data.iloc[:, feature])
            x_data.iloc[:, feature] = le.transform(x_data.iloc[:, feature])
            categorical_names[feature] = [str(x) for x in le.classes_]

        for features in categorical_features:
            try:
                tab = list(set(x_data.iloc[:,features]))
            except ValueError:
                tab = [i for i in range(len(categorical_names[features]))]
            if 0 not in tab:
                tab.insert(0, 0)
            categorical_values.append(tab)
        class_names = ['Underweight', 'Obese']
        x_data = x_data.values
        feature_transformations = {0: sex_map, 3: inverse_binary_map, 4: inverse_binary_map, 
                                   5: vegetable_map, 6: main_meal_map, 7: caec_map, 
                                   8: inverse_binary_map, 9: water_map, 10: inverse_binary_map, 
                                   11: physical_map, 12: smartphone_map, 13: caec_map, 14: mtrans_map}
        modify_feature_name = modify_obesity_feature()
    
    elif 'compas' in dataset_name:
        dataset = utils.load_dataset("compas", balance=False, discretize=False, dataset_folder="./dataset/")
        dataframe = pd.DataFrame(dataset.data, columns=dataset.feature_names)
        x_data, y_data = dataset.train, dataset.labels_train
        categorical_features = dataset.categorical_features
        tab = [i for i in range(len(dataset.train[0]))]
        categorical_names = dataset.categorical_names
        categorical_values =[]
        for feature in categorical_features:
            try:
                tab = list(set(x_data[:,feature]))
            except ValueError:
                tab = [i for i in range(len(dataset.categorical_names[feature]))]
            if not 0 in tab:
                tab.insert(0, 0)
            categorical_values.append(tab)

        class_names = ['Vanish', 'Recidiv']
        transformations = dataset.transformations

        def race_transformation(target):
            try:
                return str(dataset.race_map[target])
            except KeyError:
                return target

        def charge_degree_map_transformation(target):
            try:
                return "minor offenses" if str(dataset.charge_degree_map[target]) == "M" else "major offenses"
                #return str(dataset.charge_degree_map[target])
            except KeyError:
                return "minor offenses" if target == "M" else "major offenses"
                #return target
                    
        def charge_desc_map_transformation(target):
            try:
                return str(dataset.charge_desc_map[target])
            except KeyError:
                return target

        def temp_dont_change(target):
            return str(target)

        feature_transformations = {0: sex_map, 2:race_transformation, 
                                   3:temp_dont_change, 4:temp_dont_change, 
                                   5:temp_dont_change, 6:charge_degree_map_transformation, 
                                   7:charge_desc_map_transformation}
        modify_feature_name = modify_compas_feature()

    else:
        raise("The dataset chosen is not correct")
        
    if "obesity" not in dataset_name:
        feature_names = dataset.feature_names
    
    continuous_features = [x for x in range(len(x_data[0])) if x not in categorical_features]

    if dataframe is None:
        dataframe = pd.DataFrame(x_data, columns=feature_names)
        
    return x_data, y_data, class_names, continuous_features, \
        categorical_features, categorical_values, categorical_names, \
            feature_names, transformations, dataframe, feature_transformations, \
                modify_feature_name

def prepare_obesity_dataset():
    obesity = pd.read_csv("./dataset/obesity_original.csv")
    to_modify = {}
    binary_columns = ['family_history_with_overweight', 'FAVC', 'SMOKE','SCC']
    for binary_column in binary_columns:
        obesity = replace_binary_values(obesity, binary_column)
    to_modify['Gender'] = [['Female', 'Male'], [0, 1]]
    to_modify['CAEC'] = [['no', 'Sometimes', 'Frequently', 'Always'], [0, 1, 2, 3]]
    to_modify['CALC'] = [['no', 'Sometimes', 'Frequently', 'Always'], [0, 1, 2, 3]]
    to_modify['MTRANS'] = [['Walking', 'Bike', 'Public_Transportation', 'Automobile', 'Motorbike'], [0, 1, 2, 3, 4]]

    target = [['Insufficient_Weight', 'Normal_Weight', 'Overweight_Level_I', \
        'Overweight_Level_II', 'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III'], \
        [0, 0, 0, 0, 1, 1, 1]]
    obesity = replace_value_from_dict(obesity, to_modify)
    obesity = replace_target_value_to_binary(obesity, target)
    obesity.drop(['Weight'], axis=1, inplace=True)
    obesity.to_csv("./dataset/obesity.csv")

def replace_value_from_dict(dataframe, dictionnaire):
    for cle in dictionnaire.keys():
        dataframe[cle] = dataframe[cle].replace(dictionnaire[cle][0], dictionnaire[cle][1])
    return dataframe

def replace_binary_values(dataframe, column):
    dataframe[column] = dataframe[column].replace(['yes', 'no'], [0, 1])
    return dataframe

def replace_target_value_to_binary(dataframe, target):
    dataframe[dataframe.columns[-1]] = dataframe[dataframe.columns[-1]].replace(target[0], target[1])
    return dataframe

def sex_map(target_value):
    return 'Female' if target_value == 0 else 'Male'

def binary_map(target_value):
    target_value = int(float(target_value))
    return "Yes" if target_value == 0 else "No" 

def inverse_binary_map(target_value):
    target_value = int(float(target_value))
    return "No" if target_value == 0 else "Yes" 

def caec_map(target_value):
    target_value = int(float(target_value))
    if target_value == 0:
        return 'No'
    elif target_value == 1: 
        return 'Sometimes'
    elif target_value == 2: 
        return 'Frequently' 
    else: 
        return 'Always'

def vegetable_map(target_value):
    target_value = int(float(target_value))
    if target_value == 0:
        return "Never"
    elif target_value == 1:
        return "Sometimes"
    else:
        return "Always"

def main_meal_map(target_value):
    target_value = int(float(target_value))
    if target_value == 0:
        return 'Between 1 and 2'
    elif target_value == 1:
        return '3'
    else: 
        return "More than 3"

def water_map(target_value):
    target_value = int(float(target_value))
    if target_value == 0:
        return "Less than a liter"
    elif target_value == 1:
        return "Between 1 and 2L"
    else:
        return "More than 2L"

def physical_map(target_value):
    target_value = int(float(target_value))
    if target_value == 0:
        return "I do not have"
    elif target_value == 1:
        return "1 or 2 days"
    elif target_value == 2:
        return "2 or 4 days"
    else:
        return "4 or 5 days"

def smartphone_map(target_value):
    target_value = int(float(target_value))
    if target_value == 0:
        return "0-2 hours"
    elif target_value == 1:
        return "3-5 hours"
    else:
        return "More than 5 hours"

def mtrans_map(target_value): 
    target_value = int(float(target_value))
    if target_value == 0: 
        return 'Walking'
    elif target_value == 1: 
        return 'Bike'
    elif target_value == 2: 
        return 'Public transportation'
    elif target_value == 3: 
        return 'Automobile'
    else:
        return 'Motorbike'

def modify_obesity_feature():
    feature_name_replacement = {}
    feature_name_replacement['Gender'] = ['Gender', 0]#["What is your gender?", 0]
    feature_name_replacement['Age'] = ['Age', 1]#["What is your age?", 1]
    feature_name_replacement['Height'] = ['Height', 2]#["What is your height?", 2]
    feature_name_replacement['family_history_with_overweight'] = ['Family member has overweight', 3]#["Has a family member suffered or suffers from overweight?", 3]
    feature_name_replacement['FAVC'] = ['Frequent consumption of high caloric food', 4]#["Do you eat high caloric food frequently?", 4]
    feature_name_replacement['FCVC'] = ['Frequency of consumption of vegetables', 5]#["Do you usually eat vegetables in your meals?", 5]
    feature_name_replacement['NCP'] = ['Number of daily meals', 6]#["How many main meals do you have daily?", 6]
    feature_name_replacement['CAEC'] = ['Consumption of food between meals', 7]#["Do you eat any food between meals?", 7]
    feature_name_replacement['SMOKE'] = ['Smoke', 8]#["Do you smoke?", 8]
    feature_name_replacement['CH2O'] = ['Consumption of water daily', 9]#["How much water do you drink daily?", 9]
    feature_name_replacement['SCC'] = ['Calories consumption monitoring', 10]#["Do you monitor the calories you eat daily?", 10]
    feature_name_replacement['FAF'] = ['Physical activity frequency per week', 11]#["How often do you have physical activity?", 11]
    feature_name_replacement['TUE'] = ['Time using technology devices daily', 12]#["How much time do you use technological devices such as cell phone, videogames, television, computer and others?", 12]
    feature_name_replacement['CALC'] = ['Consumption of alcohol', 13]#["how often do you drink alcohol?", 13]
    feature_name_replacement['MTRANS'] = ['Transportation used', 14]#["Which transportation do you usually use?", 14]
    return feature_name_replacement

def modify_compas_feature():
    feature_name_replacement = {}
    feature_name_replacement['sex'] = ['Gender', 0]
    feature_name_replacement['age'] = ['Age', 1]
    feature_name_replacement['race'] = ['Race', 2]
    feature_name_replacement['juv_fel_count'] = ['Number of juvenile major offenses', 3] # nombre de crimes
    feature_name_replacement['juv_misd_count'] = ['Number of juvenile minor offenses', 4] # nombre de délits
    feature_name_replacement['priors_count'] = ['Number of previous arrest', 5] # Nombre d'antécédents
    feature_name_replacement['c_charge_degree'] = ['The degree of the charge', 6] # Le degrée d'accusation
    feature_name_replacement['c_charge_desc'] =	['Description of the charge', 7] # Description de la charge
    return feature_name_replacement
    
def transform_target_class(prediction, class_names):
    compas = "Recidiv" in class_names[1]
    if compas:
        if prediction < 0.25:
            return "No Risk"
        elif prediction < 0.5:
            return "Low Risk"
        elif prediction < 0.75:
            return "Medium Risk"
        else:
            return "High Risk"
    else:
        if prediction < 0.25:
            return "Underweight"
        elif prediction < 0.5:
            return "Healthy"
        elif prediction < 0.75:
            return "Overweight"
        else:
            return "Obesity"

def prepare_compas_dataset():
    data = pd.read_csv("./dataset/compas.csv")

    try:
        data.drop(["decile_score", 'juv_other_count', 'days_b_screening_arrest', 
                   'c_days_from_compas', 'is_recid', 'is_violent_recid', 'decile_score.1', 
                   'v_decile_score', 'priors_count.1'], axis=1, inplace=True)
    except KeyError:
        try:
            data.drop(['Unnamed: 0'], axis=1, inplace=True)
        except KeyError:
            print()
    data['two_year_recid'] = np.where(data['two_year_recid']==0, 1, 0)
    data = data.groupby('c_charge_desc').filter(lambda x : len(x)>4)
    print(data)
    data.to_csv("./dataset/compas.csv", index=False)

def generate_compas_charge_description():
    data = pd.read_csv("./dataset/compas.csv")
    charge_description = set(data['c_charge_desc'].tolist())
    print(charge_description)
    charge_desc_map = {}
    for i, charge in enumerate(charge_description):
        charge_desc_map[str(i)] = charge
    print(charge_desc_map)
    return charge_desc_map

def round_obesity_dataset():
    data = pd.read_csv("./dataset/obesity.csv")
    data['FCVC'] = data['FCVC'].round()
    data['NCP'] = data['NCP'].round()
    data['CH2O'] = data['CH2O'].round()
    data['FAF'] = data['FAF'].round()
    data['TUE'] = data['TUE'].round()
    data['Age'] = data['Age'].round()
    data['Height'] = data['Height'].round(2) * 100
    data.to_csv("./dataset/obesity.csv", index=False)

if __name__ == "__main__":
    prepare_obesity_dataset()
    prepare_compas_dataset()
    generate_compas_charge_description()
    round_obesity_dataset()