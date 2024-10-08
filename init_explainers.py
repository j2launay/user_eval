import copy
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from growingspheres.utils.gs_utils import distances
from prepare_dataset import transform_target_class

# Explanations module
from anchor import anchor_tabular
from limet import lime_tabular
from growingspheres import counterfactuals as cf
        
class InitExplainers(object): 
    def __init__(self, train_data, class_names, black_box_predict, 
                 black_box_predict_proba=None, continuous_features=None, 
                 categorical_features=None, categorical_values=None, 
                 feature_names=None, categorical_names=None, 
                 nb_min_instance_per_class_in_field=100, verbose=False, 
                 transformations=None, feature_transformations=None):
        self.verbose = verbose
        # Set all the information about the dataset that is used by the explanation modules
        self.feature_names = feature_names
        self.class_names = class_names
        self.categorical_names = categorical_names
        self.continuous_features = continuous_features
        self.categorical_features = categorical_features
        self.categorical_values = categorical_values
        self.feature_names = feature_names
        
        # We split the input data to train the explanation module on a different set of data
        self.train_data, self.test_data = train_test_split(train_data, test_size=0.4, random_state=42)
        self.max_mahalanobis = None
        test_mahanalobis = distances(self.train_data, self.train_data, self, metrics="mahalanobis")
        self.max_mahalanobis = max(test_mahanalobis['mahalanobis'])

        self.black_box_predict = lambda x: black_box_predict(x)
        # black box predict proba is used for lime explanation with probabilistic function
        if black_box_predict_proba is not None:
            self.black_box_predict_proba = lambda x: black_box_predict_proba(x)
        self.black_box_labels = black_box_predict(self.train_data)

        self.nb_min_instance_per_class_in_field = nb_min_instance_per_class_in_field
        self.feature_transformations = feature_transformations
        self.anchor_explainer = anchor_tabular.AnchorTabularExplainer(class_names, feature_names, self.train_data, 
                                                                    copy.copy(categorical_names))
        
        if categorical_features != []:
            # We fit a one hot encoder in order to generate linear explanation over one hot encoded data
            self.transformations = transformations
            self.enc = OneHotEncoder(handle_unknown='ignore')
            train_enc = self.train_data[:,categorical_features]
            self.enc.fit(train_enc)
            codes = self.enc.transform(train_enc).toarray()
            categorical_features_names = []
            for i in categorical_features:
                categorical_features_names.append(feature_names[i])
            oec_train_data = np.append(np.asarray(codes), self.train_data[:,continuous_features], axis=1)
            self.encoded_features_names = self.enc.get_feature_names_out(categorical_features_names)
            lime_features_names = []
            lime_categorical_features = []
            for i in range(len(oec_train_data[0])):
                if i < len(continuous_features):
                    lime_features_names.append(feature_names[continuous_features[i]])
                else:
                    lime_categorical_features.append(i)
            self.encoded_features_names = np.append(lime_features_names,[x for x in self.encoded_features_names]) .tolist()
            
        self.linear_explainer = lime_tabular.LimeTabularExplainer(self.train_data, feature_names=feature_names, 
                                                                categorical_features=categorical_features, 
                                                                categorical_names=categorical_names,
                                                                class_names=class_names, discretize_continuous=False,
                                                                training_labels=self.black_box_labels, 
                                                                sample_around_instance=True, kernel_width=0.2)                                                         

        # Compute and store variance of each feature
        self.feature_variance = []
        for feature in range(len(self.train_data[0])):
            self.feature_variance.append(np.std(self.train_data[:,feature]))
        # Compute and store the probability of each value for each categorical feature
        self.probability_categorical_feature = []
        if self.categorical_features is not None:
            for nb_feature, feature in enumerate(self.categorical_features):
                set_categorical_value = categorical_values[nb_feature]
                probability_instance_per_feature = []
                #sum_not_in_categorical_value = 0
                for categorical_feature in set_categorical_value:
                    probability_instance_per_feature.append(sum(self.train_data[:,feature] == categorical_feature)/len(self.train_data[:,feature]))
                self.probability_categorical_feature.append(probability_instance_per_feature)
        # We store min, max and mean values of each features in order to generate and evaluate the distance of instances according to the distribution
        self.min_features = []
        self.max_features = []
        self.mean_features = []
        continuous_features = [x for x in set(range(self.train_data.shape[1])).difference(categorical_features)]
        for continuous_feature in continuous_features:
            self.mean_features.append(np.mean(self.train_data[:,continuous_feature]))
            self.max_features.append(max(self.train_data[:,continuous_feature]))
            self.min_features.append(min(self.train_data[:,continuous_feature]))
        
        self.growing_field = cf.CounterfactualExplanation(self.black_box_predict, 
                                                    continuous_features=self.continuous_features, 
                                                    categorical_features=self.categorical_features, 
                                                    categorical_values=self.categorical_values,
                                                    max_features=self.max_features,
                                                    min_features=self.min_features,
                                                    feature_variance=self.feature_variance,  
                                                    probability_categorical_feature=self.probability_categorical_feature)

    def predict(self, instance):
        self.target_class = self.black_box_predict(instance.reshape(1, -1))[0]
        self.target_class_name = transform_target_class(self.black_box_predict_proba(instance.reshape(1, -1))[0][1], self.class_names)
        
        # Computes the distance to the farthest instance from the training dataset to bound generating instances in Growing Fields
        farthest_distance = 0
        for training_instance in self.train_data:
            farthest_distance_now = distances(training_instance, instance, self, metrics="mahalanobis")
            if farthest_distance_now > farthest_distance:
                farthest_distance = farthest_distance_now

        if self.verbose:print("### Searching for anchor explanation")
        anchor_exp = self.anchor_explainer.explain_instance(instance, self.black_box_predict, 
                                    delta=0.1, tau=0.15, batch_size=100, max_anchor_size=None, 
                                    stop_on_first=False, desired_label=None, beam_size=4)
        
        if self.verbose:print("### Searching for lime explanation")
        lime = self.linear_explainer.explain_instance(instance, self.black_box_predict_proba, 
                                                                    num_features=len(instance),
                                                                    labels=(0,1))

        if self.verbose:print("### Searching for counterfactual explanation")
        self.growing_field.fit(instance, sparse=True, verbose=self.verbose, 
                                                    farthest_distance_training_dataset=farthest_distance,
                                                    min_counterfactual_in_sphere=self.nb_min_instance_per_class_in_field)
        
        self.closest_counterfactual = self.growing_field.enemy
        self.counterfactual_class_name = transform_target_class(self.black_box_predict_proba(self.closest_counterfactual.reshape(1, -1))[0][1], 
                                                                self.class_names)
        
        return anchor_exp, lime, self.closest_counterfactual