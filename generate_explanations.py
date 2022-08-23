from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, SGDRegressor
from sklearn.naive_bayes import GaussianNB
from prepare_dataset import generate_dataset, preparing_dataset
from init_explainers import InitExplainers
import warnings
import os
import pandas as pd

if __name__ == "__main__":
    # Filter the warning from matplotlib
    warnings.filterwarnings("ignore")
    # Datasets used for the experiments 
    dataset_names = ["diabetes"]#"adult"]#"titanic"]#"compas", "cancer"]#"generate_blob"]#"generate_circles"]#"generate_moons"]
    #'categorical_generate_blobs']#"generate_blobs"]#"blood"]#"cancer"]#"compas"]#"adult"]#"diabetes"]#'titanic']#
    
    models = [GaussianNB(),
                VotingClassifier(estimators=[('lr', LogisticRegression()), ('gnb', GaussianNB()), ('svm', svm.SVC(probability=True))], voting='soft'),#('rc', RidgeClassifier())], voting="soft"),
                GradientBoostingClassifier(n_estimators=20, learning_rate=1.0, random_state=1),
                RandomForestClassifier(n_estimators=20, random_state=1), 
                MLPClassifier(random_state=1, activation='logistic'),
                svm.SVC(probability=True, random_state=1, class_weight="balanced")]

    # Number of instances explained by each model on each dataset
    max_instance_to_explain = 40
    """ All the variable necessaries for generating the graph results """
    # Store results inside graph if set to True
    graph = True
    verbose = False
    growing_sphere = False
    growing_method = "GS" if growing_sphere else "GF"
    distance_metric = "mahalanobis"

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
                    feature_names, transformations, dataframe = generate_dataset(dataset_name)
        
        print(dataframe.head())

        for nb_model, black_box in enumerate(models):
            model_name = type(black_box).__name__
            if growing_sphere:
                filename = "./results/"+dataset_name+"/growing_spheres/"+model_name+"/"+str(threshold_interpretability)+"/" + linear_model_name
                filename_all = "./results/"+dataset_name+"/growing_spheres/"+str(threshold_interpretability)+"/" + linear_model_name
            else:
                filename="./results/"+dataset_name+"/"+model_name+"/"+str(threshold_interpretability)+"/" + linear_model_name
                filename_all="./results/"+dataset_name+"/"+str(threshold_interpretability)+"/" + linear_model_name
            models_name.append(model_name)
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
                                                            transformations=transformations)
            
            temp_counterfactuals, temp_rules = pd.DataFrame(columns=dataframe.columns), pd.DataFrame()#[], []
            os.makedirs(os.path.dirname("./results/" + dataset_name + "/" + model_name + "/"), exist_ok=True)

            for instance_to_explain in x_test:
                if cnt == max_instance_to_explain:
                    break
                print("### Instance number:", cnt + 1, "over", max_instance_to_explain)
                print("### Models ", nb_model + 1, "over", len(models))
                print("instance to explain:", instance_to_explain)

                try:
                    #test += 2
                #except NameError:
                    anchor_exp, lime, closest_counterfactual, local_surrogate = explainers.predict(instance_to_explain, 
                                                                                linear_model=linear_model, 
                                                                                distance_metric=distance_metric,
                                                                                nb_features_employed=6)
                    """print("rule by anchor", anchor_exp.names())
                    print()
                    print("rule by lime:")
                    print('\n'.join(map(str, lime.as_list())))
                    print()
                    print("rule by local surrogate:")
                    print('\n'.join(map(str, local_surrogate.as_list())))
                    print()
                    print("Growing Fields closest counterfactual:")
                    print(closest_counterfactual)"""

                    dataframe.loc[dataframe.shape[0]] = instance_to_explain
                    dataframe.loc[dataframe.shape[0]] = closest_counterfactual
                    dataframe.loc[dataframe.shape[0]] = closest_counterfactual-instance_to_explain
                    counterfactuals = dataframe.tail(3)
                    counterfactuals.rename(index={dataframe.shape[0]-3:'target ' + str(str(cnt)), dataframe.shape[0]-2:'counterfactual ' + str(cnt), dataframe.shape[0]-1:'explanation ' + str(cnt)}, inplace=True)
                    temp_counterfactuals = pd.concat([temp_counterfactuals, counterfactuals])
                    counterfactuals.to_csv("./results/" + dataset_name + "/" + model_name + "/counterfactual_explanations.csv")
                    
                    anchor_rule = [anchor_exp.names()]
                    lime_rule = '\n'.join(map(str, lime.as_list()))
                    local_surrogate_rule = '\n'.join(map(str, local_surrogate.as_list()))
                    explanations_rule = pd.DataFrame({'anchor':anchor_rule, 'lime':lime_rule, 'local surrogate':local_surrogate_rule})
                    explanations_rule.to_csv("./results/" + dataset_name + "/" + model_name + "/rules_explanations.csv")
                    temp_rules = pd.concat([temp_rules, explanations_rule])
                    
                    temp_counterfactuals.to_csv("./results/" + dataset_name + "/" + model_name + "/counterfactual_explanations.csv")
                    temp_rules.to_csv("./results/" + dataset_name + "/" + model_name + "/rules_explanations.csv")
                    cnt += 1
                except Exception as inst:
                    print(inst)
                if cnt %5 == 0:
                    print()
                    print("### Instance number:", cnt , "over", max_instance_to_explain, 'with', model_name, 'on', dataset_name)


            