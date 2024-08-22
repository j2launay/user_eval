from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from prepare_dataset import generate_dataset
from init_explainers import InitExplainers
from VisualisationExplanation import VisualisationExplanation
import warnings
import os
import pandas as pd

if __name__ == "__main__":
    # Filter the warning from matplotlib
    warnings.filterwarnings("ignore")
    # Datasets used for the experiments 
    dataset_names = ["compas", "obesity"]
    # Black box model to explain
    models = [#GaussianNB(), 
        MLPClassifier(random_state=1, activation='logistic')]

    # Number of instances explained by each model on each dataset
    max_instance_to_explain = 75
    verbose = False

    visualisation_explanation = VisualisationExplanation()

    # Initialize all the variable needed to store the result in graph
    for dataset_name in dataset_names:
        models_name = []
        # Store dataset inside x and y (x corresponds to the data and y to the labels), with aditional information
        x, y, class_names, continuous_features, categorical_features, \
            categorical_values, categorical_names, feature_names, transformations, \
                dataframe, feature_transformations, modify_feature_name = \
                    generate_dataset(dataset_name)
        
        print(dataframe.head())
        counterfactuals_columns = dataframe.columns
        counterfactuals_columns += 'target'
        for nb_model, black_box in enumerate(models):
            model_name = type(black_box).__name__
            models_name.append(model_name)
            filename = "./results/" + dataset_name + "/" + model_name + "/"
            # Split the dataset into training and testing set (70% training and 30% test)
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=10)
            print("###", model_name, "training on", dataset_name, "dataset.")
            print()
            black_box = black_box.fit(x_train, y_train)
            print('### Accuracy:', black_box.score(x_test, y_test))
            cnt = 0
            
            # Initialisation of the various explanation techniques
            explainers = InitExplainers(x_train, class_names, black_box.predict, 
                                        black_box.predict_proba, 
                                        continuous_features=continuous_features,
                                        categorical_features=categorical_features, 
                                        categorical_values=categorical_values, 
                                        feature_names=feature_names, 
                                        categorical_names=categorical_names,
                                        transformations=transformations, 
                                        feature_transformations=feature_transformations)
            temp_counterfactuals, temp_rules = pd.DataFrame(columns=counterfactuals_columns), pd.DataFrame()

            for instance_to_explain, label in zip(x_test, y_test):
                if cnt == max_instance_to_explain:
                    break
                
                os.makedirs(os.path.dirname("./results/" + dataset_name + "/" + model_name + "/" + str(cnt) + "/"), exist_ok=True)
                filename_per_instance = "./results/" + dataset_name + "/" + model_name + "/" + str(cnt) + "/"
                explainers.filename = filename_per_instance
                print("### Instance number:", cnt + 1, "over", max_instance_to_explain, 'with', model_name, 'on', dataset_name)
                print("### Models ", nb_model + 1, "over", len(models))
                print("instance to explain:", instance_to_explain)

                try:
                    rule_explanation, linear_explanation, closest_counterfactual = explainers.predict(instance_to_explain)

                    # Feature attribution visualisation
                    lime_normalised, sum_coef_lime = visualisation_explanation.normalize_linear_explanation(linear_explanation, 
                                                                                                            black_box.predict_proba(instance_to_explain.reshape(1, -1))[0][1])
                    lime_rule = '\n'.join(map(str, linear_explanation.as_list(label)))
                    pos_lime_exp, neg_lime_exp, other_features_sum_values = visualisation_explanation.generate_linear_text_explanation(lime_normalised, 
                                                modify_feature_name, filename_per_instance, instance_to_explain.copy(), categorical_features, feature_transformations, 
                                                sum_coef_lime, explainers)
                    visualisation_explanation.generate_linear_image_explanation(pos_lime_exp, neg_lime_exp, filename_per_instance, explainers, other_features_sum_values)
                    visualisation_explanation.generate_target_instance_array(explainers, modify_feature_name, filename_per_instance, instance_to_explain)
                    
                    # Counterfactual representation
                    counterfactual_text_representation, initial_prediction_superior = visualisation_explanation.generate_counterfactual_text(explainers, 
                                                                                closest_counterfactual, instance_to_explain.copy())
                    visualisation_explanation.generate_counterfactual_image(counterfactual_text_representation, modify_feature_name, filename_per_instance, 
                                                                                explainers, initial_prediction_superior)

                    # Rule based representation
                    anchor_rule = visualisation_explanation.generate_anchor_image(rule_explanation, modify_feature_name, categorical_features, 
                                                                                  feature_transformations, filename_per_instance)
                    
                    cnt += 1
                except Exception as inst:
                    print(inst)
