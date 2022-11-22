from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, SGDRegressor
from sklearn.naive_bayes import GaussianNB
from prepare_dataset import generate_dataset, preparing_dataset
from init_explainers import InitExplainers
from VisualisationExplanation import VisualisationExplanation
import warnings
import os
import pandas as pd

if __name__ == "__main__":
    # Filter the warning from matplotlib
    warnings.filterwarnings("ignore")
    # Datasets used for the experiments 
    dataset_names = ["compas"]#"obesity"]#"blood"]#"adult"]#"mortality"]#"heart"]#diabetes"]#"adult"]#"compas"]#"cancer"]#"titanic"]#
    
    models = [#GaussianNB(),
                GradientBoostingClassifier(n_estimators=20, learning_rate=1.0, random_state=1),
                MLPClassifier(random_state=1, activation='logistic'),
                RandomForestClassifier(n_estimators=20, random_state=1), 
                svm.SVC(probability=True, random_state=1, class_weight="balanced"),
                VotingClassifier(estimators=[('lr', LogisticRegression()), ('gnb', GaussianNB()), ('svm', svm.SVC(probability=True))], voting='soft')#('rc', RidgeClassifier())], voting="soft"),
                ]

    # Number of instances explained by each model on each dataset
    max_instance_to_explain = 75
    """ All the variable necessaries for generating the graph results """
    # Store results inside graph if set to True
    graph = True
    verbose = False
    growing_sphere = False
    growing_method = "GS" if growing_sphere else "GF"
    distance_metric = "mahalanobis"

    visualisation_explanation = VisualisationExplanation()
    # Threshold for explanation method precision
    threshold_interpretability = 0.95
    linear_separability_index = 1
    linear_model = Ridge(alpha=0)#SGDRegressor()#LinearRegression()#
    linear_model_name = type(linear_model).__name__ + "_" if "Ridge" not in type(linear_model).__name__ else ""

    # Initialize all the variable needed to store the result in graph
    for dataset_name in dataset_names:
        models_name = []
        # Store dataset inside x and y (x data and y labels), with aditional information
        x, y, class_names, regression, multiclass, continuous_features, categorical_features, categorical_values, categorical_names, \
                    feature_names, transformations, dataframe, feature_transformations, modify_feature_name, lime_feature_name = generate_dataset(dataset_name)
        
        print(dataframe.head())

        for nb_model, black_box in enumerate(models):
            model_name = type(black_box).__name__
            models_name.append(model_name)
            filename = "./results/" + dataset_name + "/" + model_name + "/"
            # Split the dataset inside train and test set (70% training and 30% test)
            x_train, x_test, y_train, y_test = preparing_dataset(x, y)
            print("###", model_name, "training on", dataset_name, "dataset.")
            print()
            print()
            black_box = black_box.fit(x_train, y_train)
            print('### Accuracy:', black_box.score(x_test, y_test))
            cnt = 0
                
            explainers = InitExplainers(x_train, class_names, black_box.predict, black_box.predict_proba,
                                                            growing_method=growing_method,
                                                            continuous_features=continuous_features,
                                                            categorical_features=categorical_features, categorical_values=categorical_values, 
                                                            feature_names=feature_names, categorical_names=categorical_names,
                                                            verbose=verbose, threshold_precision=threshold_interpretability,
                                                            linear_separability_index=linear_separability_index, 
                                                            transformations=transformations, feature_transformations=feature_transformations)
            counterfactuals_columns = dataframe.columns 
            counterfactuals_columns += 'target'
            temp_counterfactuals, temp_rules = pd.DataFrame(columns=counterfactuals_columns), pd.DataFrame()#[], []

            for instance_to_explain, label in zip(x_test[6:], y_test[6:]):
                if cnt == max_instance_to_explain:
                    break
                os.makedirs(os.path.dirname("./results/" + dataset_name + "/" + model_name + "/" + str(cnt) + "/"), exist_ok=True)
                print("### Instance number:", cnt + 1, "over", max_instance_to_explain)
                print("### Models ", nb_model + 1, "over", len(models))
                print("instance to explain:", instance_to_explain)
                print("its associated proba", black_box.predict_proba(instance_to_explain.reshape(1, -1)))

                try:
                    #test += 2
                #except NameError:
                    filename_per_instance = "./results/" + dataset_name + "/" + model_name + "/" + str(cnt) + "/"
                    explainers.filename = filename_per_instance
                    anchor_exp, lime, closest_counterfactual, local_surrogate = explainers.predict(instance_to_explain, 
                                                                                linear_model=linear_model, 
                                                                                distance_metric=distance_metric,
                                                                                nb_features_employed=None)#6)

                    lime_normalised = visualisation_explanation.normalize_linear_explanation(lime, black_box.predict_proba(instance_to_explain.reshape(1, -1))[0][1])
                    dataframe.loc[dataframe.shape[0]] = instance_to_explain
                    dataframe.loc[dataframe.shape[0]] = closest_counterfactual
                    dataframe.loc[dataframe.shape[0]] = closest_counterfactual-instance_to_explain
                    counterfactuals = dataframe.tail(3)
                    counterfactuals.rename(index={dataframe.shape[0]-3:'target ' + str(str(cnt)), dataframe.shape[0]-2:'counterfactual ' + str(cnt), \
                        dataframe.shape[0]-1:'explanation ' + str(cnt)}, inplace=True)
                    counterfactuals['prediction'] = class_names[black_box.predict(instance_to_explain.reshape(1, -1))[0]]
                    counterfactuals['target'] = explainers.target_class_name
                    temp_counterfactuals = pd.concat([temp_counterfactuals, counterfactuals])
                    counterfactuals.to_csv("./results/" + dataset_name + "/" + model_name + "/counterfactual_explanations.csv")
                    

                    # LIME visualisation
                    lime_rule = '\n'.join(map(str, lime.as_list()))
                    pos_lime_exp, neg_lime_exp, other_features_sum_values = visualisation_explanation.generate_linear_text_explanation(lime_normalised, 
                                                modify_feature_name, filename_per_instance, instance_to_explain.copy(), categorical_features, feature_transformations, 
                                                black_box.predict_proba(instance_to_explain.reshape(1, -1))[0][1], explainers)
                    visualisation_explanation.generate_linear_image_explanation(pos_lime_exp, neg_lime_exp, filename_per_instance, explainers, other_features_sum_values)
                    
                    visualisation_explanation.generate_target_instance_array(explainers, modify_feature_name, filename_per_instance, instance_to_explain)
                    
                    # Growing Fields representation
                    counterfactual_text_representation, initial_prediction_superior = visualisation_explanation.generate_counterfactual_text(explainers, \
                                                                                closest_counterfactual, instance_to_explain.copy())
                    visualisation_explanation.generate_counterfactual_image(counterfactual_text_representation, modify_feature_name, filename_per_instance, \
                                                                                explainers, initial_prediction_superior)

                    # Anchors representation
                    anchor_rule = visualisation_explanation.generate_anchor_image(anchor_exp, modify_feature_name, categorical_features, feature_transformations, \
                                                                                filename_per_instance, explainers)

                    """print(anchor_exp.names())
                    print("and the associate precision for each sub rule:")
                    print(anchor_exp.exp_map['precision'])"""



                    local_surrogate_rule = '\n'.join(map(str, local_surrogate.as_list()))
                    explanations_rule = pd.DataFrame({'anchor':anchor_rule, 'lime':lime_rule, 'local surrogate':local_surrogate_rule, \
                        'prediction':class_names[black_box.predict(instance_to_explain.reshape(1, -1))[0]] + "prob:" + \
                            str(black_box.predict_proba(instance_to_explain.reshape(1, -1))), 'target': class_names[label]})
                    explanations_rule.to_csv(filename + "rules_explanations.csv")
                    temp_rules = pd.concat([temp_rules, explanations_rule])
                    
                    temp_counterfactuals.to_csv(filename + "counterfactual_explanations.csv")
                    temp_rules.to_csv(filename + "rules_explanations.csv")

                    lime.save_to_file(filename_per_instance + "lime_explanation.html")
                    anchor_exp.save_to_file(filename_per_instance + "anchor_explanation.html")
                    explanations_rule.to_csv(filename_per_instance + "rules_explanations.csv")
                    cnt += 1
                except Exception as inst:
                    print(inst)
                if cnt %5 == 0:
                    print()
                    print("### Instance number:", cnt , "over", max_instance_to_explain, 'with', model_name, 'on', dataset_name)
