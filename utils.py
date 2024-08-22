import copy
import sklearn
import numpy as np
import pandas as pd
import limet
import os
os.environ['SPACY_WARNING_IGNORE'] = 'W008'
import sys

if (sys.version_info > (3, 0)):
    def unicode(s, errors=None):
        return s#str(s)

class Bunch(object):
    """bla"""
    def __init__(self, adict):
        self.__dict__.update(adict)


def map_array_values(array, value_map):
    # value map must be { src : target }
    ret = array.copy()
    for src, target in value_map.items():
        ret[ret == src] = target
    return ret
def replace_binary_values(array, values):
    return map_array_values(array, {'0': values[0], '1': values[1]})

def load_dataset(dataset_name, balance=False, discretize=True, dataset_folder='./', X=None, y=None, plot=False):
    if "obesity" in dataset_name:
        feature_names = ['Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight',
                'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE', 'CALC', 'MTRANS', 'NObeyesdad']
        dataset = load_csv_dataset(
            os.path.join(dataset_folder, 'obesity.csv'), -1, ', ',
            feature_names=feature_names, features_to_use=features_to_use,
            categorical_features=categorical_features, 
            discretize=discretize,
            balance=balance, feature_transformations=transformations)
        dataset.class_names = ['Normal', 'Obese']
        dataset.transformations = transformations
    
    elif dataset_name == 'compas':
        categorical_features = [0, 2, 3, 4, 5, 6, 7]
        transformations = {
            0: lambda x: map_array_values(x, sex_map),
            2: lambda x: map_array_values(x, race_map),
            6: lambda x: map_array_values(x, charge_degree_map),
            7: lambda x: map_array_values(x, charge_desc_map),
        }
        sex_map = {0: 'Female', 1: 'Male'}
        race_map = {0: 'Other', 1: "African-American", 2: 'Caucasian', 3: 'Hispanic', 4: 'Native American', 5: 'Asian'}
        charge_degree_map = {0: 'F', 1: 'M'}
        def generate_compas_charge_description(filename):
            data = pd.read_csv(filename)
            charge_description = set(data['c_charge_desc'].tolist())
            charge_desc_map = {}
            for i, charge in enumerate(charge_description):
                charge_desc_map[str(i)] = charge
            return charge_desc_map
        charge_desc_map = generate_compas_charge_description(dataset_folder + 'compas.csv')
        dataset = load_csv_dataset(
            os.path.join(dataset_folder, 'compas.csv'), -1, ',',
            categorical_features=categorical_features, 
            discretize=discretize,
            balance=balance, feature_transformations=transformations)
        dataset.class_names = ['Recidiv', 'Disappear']
        dataset.transformations = transformations
        dataset.charge_desc_map, dataset.race_map, dataset.charge_degree_map = charge_desc_map, race_map, charge_degree_map
    return dataset



def load_csv_dataset(data, target_idx, delimiter=',',
                     feature_names=None, categorical_features=None,
                     features_to_use=None, feature_transformations=None,
                     discretize=False, balance=False, fill_na='-1', filter_fn=None, skip_first=False, data_generate=None):
    """if not feature names, takes 1st line as feature names
    if not features_to_use, use all except for target
    if not categorical_features, consider everything < 20 as categorical"""
    if feature_transformations is None:
        feature_transformations = {}
    if data_generate == None:
        try:
            data = np.genfromtxt(data, delimiter=delimiter, dtype='|S128')
        except:
            import pandas
            data = pandas.read_csv(data,
                               header=None,
                               delimiter=delimiter,
                               na_filter=True,
                               dtype=str).fillna(fill_na).values
    if target_idx < 0:
        target_idx = data.shape[1] + target_idx
    ret = Bunch({})
    if feature_names is None:
        feature_names = list(data[0])
        data = data[1:]
    else:
        feature_names = copy.deepcopy(feature_names)
    if skip_first:
        data = data[1:]
    if filter_fn is not None:
        data = filter_fn(data)
    
    for feature, fun in feature_transformations.items():
        data[:, feature] = fun(data[:, feature])
    labels = data[:, target_idx]
    le = sklearn.preprocessing.LabelEncoder()
    le.fit(labels)
    ret.labels = le.transform(labels)
    labels = ret.labels
    ret.class_names = list(le.classes_)
    ret.class_target = feature_names[target_idx]
    if features_to_use is not None:
        data = data[:, features_to_use]
        feature_names = ([x for i, x in enumerate(feature_names)
                          if i in features_to_use])
        if categorical_features is not None:
            categorical_features = ([features_to_use.index(x)
                                     for x in categorical_features])
    else:
        data = np.delete(data, target_idx, 1)
        feature_names.pop(target_idx)
        if categorical_features:
            categorical_features = ([x if x < target_idx else x - 1
                                     for x in categorical_features])

    if categorical_features is None:
        categorical_features = []
        for f in range(data.shape[1]):
            if len(np.unique(data[:, f])) < 20:
                categorical_features.append(f)
    categorical_names = {}
    for feature in categorical_features:
        le = sklearn.preprocessing.LabelEncoder()
        le.fit(data[:, feature])
        data[:, feature] = le.transform(data[:, feature])
        categorical_names[feature] = le.classes_
    data = data.astype(float)
    ordinal_features = []
    if discretize:
        disc = limet.lime_tabular.QuartileDiscretizer(data,
                                                     categorical_features,
                                                     feature_names)
        data = disc.discretize(data)
        ordinal_features = [x for x in range(data.shape[1])
                            if x not in categorical_features]
        categorical_features = range(data.shape[1])
        categorical_names.update(disc.names)
    for x in categorical_names:
        categorical_names[x] = [y.decode() if type(y) == np.bytes_ else y for y in categorical_names[x]]
    ret.ordinal_features = ordinal_features
    ret.categorical_features = categorical_features
    ret.categorical_names = categorical_names
    ret.feature_names = feature_names
    np.random.seed(1)
    if balance:
        idxs = np.array([], dtype='int')
        min_labels = np.min(np.bincount(labels))
        for label in np.unique(labels):
            idx = np.random.choice(np.where(labels == label)[0], min_labels)
            idxs = np.hstack((idxs, idx))
        data = data[idxs]
        labels = labels[idxs]
        ret.data = data
        ret.labels = labels

    splits = sklearn.model_selection.ShuffleSplit(n_splits=1,
                                                  test_size=.2,
                                                  random_state=1)
    train_idx, test_idx = [x for x in splits.split(data)][0]
    ret.train = data[train_idx]
    ret.labels_train = ret.labels[train_idx]
    cv_splits = sklearn.model_selection.ShuffleSplit(n_splits=1,
                                                     test_size=.5,
                                                     random_state=1)
    cv_idx, ntest_idx = [x for x in cv_splits.split(test_idx)][0]
    cv_idx = test_idx[cv_idx]
    test_idx = test_idx[ntest_idx]
    ret.validation = data[cv_idx]
    ret.labels_validation = ret.labels[cv_idx]
    ret.test = data[test_idx]
    ret.labels_test = ret.labels[test_idx]
    ret.test_idx = test_idx
    ret.validation_idx = cv_idx
    ret.train_idx = train_idx
    ret.data = data
    return ret