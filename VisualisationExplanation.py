from mimetypes import init
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import csv
from prepare_dataset import transform_target_class

def save_dictionary_to_csv(filename, dictionary, dictionnaire=True):
    w = csv.writer(open(filename, "w"))
    if dictionnaire:
        for dict in dictionary:
            for key, val in dict.items():
                w.writerow([key, val])
            w.writerow([])
    else:
        for line in dictionary:
            w.writerow(line)

def split_explanation(feature):
    rule = feature.split("<=")
    if len(rule) == 1:
        rule = rule[0].split(">=")
    if len(rule) == 1:
        rule = rule[0].split("=")
    if len(rule) == 1:
        rule = rule[0].split("<")
    if len(rule) == 1:
        rule = rule[0].split(">")
    return rule

def anchor_image_fancy(init_explainers, filename):
    plt.title("anchor rule for " + init_explainers.target_class_name)
    plt.xticks([0], labels=['rule'])
    plt.ylim([0, 1.1])
    plt.xlim([-0.3, 0.3])
    plt.ylabel('precision')
    plt.savefig(filename + "rule_image.png")
    plt.show(block=False)
    plt.pause(0.1)
    plt.close('all')

def counterfactual_image_fancy(init_explainers, label, last_prediction, filename):
    plt.text(0, last_prediction, label, ha='center', va='bottom')
    plt.title("How to modify the prediction from " + init_explainers.target_class_name + " to " + init_explainers.counterfactual_class_name)
    plt.xticks([0], labels=['counterfactual'])
    plt.ylim([0, 1.1])
    plt.xlim([-0.3, 0.3])
    plt.ylabel('precision')
    plt.savefig(filename + "counterfactual_image.png")
    plt.show(block=False)
    plt.pause(0.1)
    plt.close('all')

def linear_image_fancy(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    left, bottom, width, height = (0, -0.5, 0.25, 5)
    rect=mpatches.Rectangle((left,bottom),width,height, 
                    alpha=0.2,
                    facecolor="blue")
    plt.gca().add_patch(rect)
    left = 0.25
    rect=mpatches.Rectangle((left,bottom),width,height, 
                    alpha=0.2,
                    facecolor="green")
    plt.gca().add_patch(rect)
    left = 0.5
    rect=mpatches.Rectangle((left,bottom),width,height, 
                    alpha=0.2,
                    facecolor="yellow")
    plt.gca().add_patch(rect)
    left = 0.75
    rect=mpatches.Rectangle((left,bottom),width,height, 
                    alpha=0.2,
                    facecolor="red")
    plt.gca().add_patch(rect)
    ax.set_xlim(0, 1)

def linear_image_fancy_end(ax, filename):
    ax.xaxis.set_tick_params(width=0)
    ax.yaxis.set_tick_params(width=0, pad=15)
    plt.subplots_adjust(left=0.32)
    #arrow = mpatches.FancyArrow(0., 0., 0.4, 0.6)
    #ax.add_line(arrow)
    plt.savefig(filename + "linear_image.png")
    plt.show(block=False)
    plt.pause(0.1)
    plt.close('all')

class VisualisationExplanation(object):
    def __init__(self):
        print()
    
    def generate_target_instance_array(self, init_explainers, modify_feature_name, filename, target_instance):
        last_prediction = str(init_explainers.black_box_predict_proba(target_instance.reshape(1, -1))[0][1].round(3))
        target_instance_text = [["Target instance classified as " + init_explainers.target_class_name + " with a probability of " + str(last_prediction)]]
        for feature_name in modify_feature_name:
            if modify_feature_name[feature_name][1] in init_explainers.categorical_features:
                answer = init_explainers.feature_transformations[modify_feature_name[feature_name][1]](target_instance[modify_feature_name[feature_name][1]])
            else:
                answer = target_instance[modify_feature_name[feature_name][1]]
            target_instance_text.append([modify_feature_name[feature_name][0], answer]) 
        save_dictionary_to_csv(filename + "target_instance.csv", target_instance_text, dictionnaire=False)

    def generate_anchor_image(self, anchor_exp, modify_feature_name, categorical_features, feature_transformations, filename_per_instance, init_explainers):
        plt.figure(figsize=(9,9))
        anchor_rule = {}
        last_precision = 0
        for rule, precision in zip(anchor_exp.names(), anchor_exp.exp_map['precision']):
            anchor_rule[rule] = precision
            rule = split_explanation(rule)
            rule[0], nb_feature = modify_feature_name[rule[0][:-1]][0], modify_feature_name[rule[0][:-1]][1]
            if nb_feature in categorical_features:
                answer = feature_transformations[nb_feature](rule[1])
            else:
                answer = rule[1]
            precision = round(precision, 2)
            label = rule[0] + " " + answer
            plt.bar(0, precision-last_precision, label=label, bottom=last_precision, width=0.3)
            plt.text(0, (precision+last_precision)/2 - 0.02, label + '\n increases precision by ' + str(round(precision-last_precision, 2)), ha='center', va='bottom')
            last_precision = precision
        save_dictionary_to_csv(filename_per_instance + "anchor_rule.csv", [anchor_rule])
        anchor_image_fancy(init_explainers, filename_per_instance)
        return anchor_rule

    def generate_counterfactual_image(self, counterfactual_exp, modify_feature_name, filename_per_instance, init_explainers):
        last_prediction = counterfactual_exp[0]['initial prediction']        
        target_instance_text = [["If you change:"]]
        plt.figure(figsize=(9, 9))
        plt.bar(0, last_prediction, label='initial prediction= ' + str(last_prediction), width=0.3)
        for counterfactual in counterfactual_exp[1:]:
            feature_translate = modify_feature_name[counterfactual['feature name']][0]
            label = feature_translate + '\n changing from ' + counterfactual['value target'] + ' to ' + \
                counterfactual['value cf'] + '\n reduces prediction of ' + str(round(last_prediction - counterfactual['prediction'], 3))
            plt.bar(0, counterfactual['prediction'], label=label, width=0.3)
            plt.text(0, (last_prediction + counterfactual['prediction']) / 2 - 0.03, label, ha='center', va='bottom')
            target_instance_text.append([feature_translate, counterfactual['value target'], "into", counterfactual['value cf']])
            last_prediction = counterfactual['prediction']
        target_instance_text.append(['the prediction would be ' + init_explainers.counterfactual_class_name])# + " with a probability of " + str(last_prediction)])
        save_dictionary_to_csv(filename_per_instance + "counterfactual_text.csv", target_instance_text, dictionnaire=False)
        label = "if everything changes the prediction is " + str(last_prediction)
        counterfactual_image_fancy(init_explainers, label, last_prediction, filename_per_instance)

    def normalize_linear_explanation(self, lime, target_predict_proba):
        sum_coefficient_features = 0
        for element in lime.as_list():
            sum_coefficient_features += element[1]
        
        normalized_coefficient_values = []
        for element in lime.as_list():
            normalized_coefficient_values.append(round(element[1]/sum_coefficient_features*target_predict_proba, 2))

        lime_exp_normalised = []
        nb_feature, old_coef = 0, 0
        for coefficient, lime_exp in zip(normalized_coefficient_values, lime.as_list()):
            temp_difference_coefficient_value = abs(old_coef) - abs(coefficient)
            if nb_feature > 1 and (nb_feature > 4 or temp_difference_coefficient_value > abs(coefficient)/2):
                break
            else:
                old_coef = coefficient
            lime_exp_normalised.append([lime_exp[0], coefficient])
            nb_feature += 1
        return lime_exp_normalised
    
    def generate_linear_image_explanation(self, pos_lime_exp, neg_lime_exp, filename, init_explainer, other_features_sum_values):
        def store_graph_values_left(feature_lime_exp, left_array_graph_value):
            # Store values from last feature to start the graph representation at the last black box prediction
            last = 0
            for feature_exp in feature_lime_exp:
                left_array_graph_value.append(feature_exp[2] + last)
                last = feature_exp[2] + last
            return left_array_graph_value
        left_array_graph_value = [0]
        left_array_graph_value = store_graph_values_left(pos_lime_exp, left_array_graph_value)
        left_array_graph_value = store_graph_values_left(neg_lime_exp, left_array_graph_value)
        
        labels = ["Other parameters"]
        labels += [x[0] for x in reversed(neg_lime_exp)]
        labels += [x[0] for x in reversed(pos_lime_exp)]
        
        plt.rcdefaults()
        fig, ax = plt.subplots(figsize=(18, 7))
        linear_image_fancy(ax)

        label_size = 18
        # Generate the label of each class prediction
        ax.set_xticks([0.12, 0.37, 0.62, 0.87], labels=[transform_target_class(0.12), transform_target_class(0.37), \
                        transform_target_class(0.62), transform_target_class(0.87)], size=label_size+2)
        
        # Generate the bar indicating the final prediction of the AI model as well as the prediction by the model
        plt.axvline(round(left_array_graph_value[-1] + other_features_sum_values, 2), color="black", linewidth=5)
        x_bounds = ax.get_xlim()
        ax.annotate(text = "Prediction: " + init_explainer.target_class_name, xy =((((left_array_graph_value[-1] + \
                other_features_sum_values)-x_bounds[0])/(x_bounds[1]-x_bounds[0])) - 0.05, 1.01), 
                    xycoords='axes fraction', color="black", size=label_size+4, weight="bold")
        
        # Color of the other features sum values
        other_color = "Blue" if other_features_sum_values < 0 else "Red"
        ax.barh([0], [round(other_features_sum_values, 2)], left=[left_array_graph_value[-1]], color=other_color)
        # Pos features from lime exp
        ax.barh(np.arange(len(neg_lime_exp) + 1, len(pos_lime_exp) + \
            len(neg_lime_exp) + 1), [x[1] for x in reversed(pos_lime_exp)], \
            left=left_array_graph_value[:len(pos_lime_exp)][::-1], color="Red", alpha=1)
        # Neg features from lime exp
        ax.barh(np.arange(1, len(neg_lime_exp) + 1), [x[1] for x in reversed(neg_lime_exp)], \
                left=left_array_graph_value[len(pos_lime_exp): len(pos_lime_exp) + len(neg_lime_exp)][::-1], 
                    color="Blue", alpha=1)
        # Generate the y labels with question and answers from the user
        ax.set_yticks(np.arange(len(pos_lime_exp) + len(neg_lime_exp) + 1), labels=labels, size=label_size)
        
        # Add the score of each feature value at the side of the bar
        labels_bar = [["+" + str(round(other_features_sum_values, 2)) if other_features_sum_values > 0 \
            else str(round(other_features_sum_values, 2))],
                ["+" + str(x[1]) for x in reversed(pos_lime_exp)], [str(x[1]) for x in reversed(neg_lime_exp)]]
        for bars, labels in zip(ax.containers, labels_bar):
            ax.bar_label(bars, labels=labels, size=label_size+2)
            
        linear_image_fancy_end(ax, filename)

    def generate_linear_text_explanation(self, lime, modify_feature_name, filename, target_instance, 
                        categorical_features, feature_transformations, target_proba, init_explainer):
        final_text = []
        sum_coef = 0
        pos_graphical_answer_question, neg_graphical_answer_question = [], []
        for lime_exp in lime:
            sum_coef += lime_exp[1]
            feature = lime_exp[0]
            rule = split_explanation(feature)
            question, nb_feature = modify_feature_name[rule[0]][0], modify_feature_name[rule[0]][1]
            if nb_feature in categorical_features:
                answer = str(feature_transformations[nb_feature](rule[1]))
            else:
                answer = str(target_instance[nb_feature])

            if lime_exp[1] > 0:
                pos_graphical_answer_question.append([question + "\n", answer, lime_exp[1]])
            else:
                neg_graphical_answer_question.append([question + "\n", answer, lime_exp[1]])
            
            text = "answering " + answer + " to " + question
            text += " increases " if lime_exp[1] > 0 else " reduces "
            text += "the risk of being \n " + init_explainer.target_class_name + " \n of " + str(round(lime_exp[1], 2))
            final_text.append([text])

        other_features_sum_values = target_proba - sum_coef
        additional_text = "the other parameters increases " if other_features_sum_values > 0 else "the other parameters reduces " 
        additional_text += "the prediction of " + str(round(other_features_sum_values, 2)) + " to be " + init_explainer.target_class_name
        final_text.append([additional_text])
        save_dictionary_to_csv(filename + "linear_text.csv", final_text, dictionnaire=False)
        return pos_graphical_answer_question, neg_graphical_answer_question, other_features_sum_values

    def generate_counterfactual_text(self, init_explainers, counterfactual, target_instance):
        target = target_instance.copy()
        counterfactual_explanation = target-counterfactual
        initial_prediction_all = init_explainers.black_box_predict_proba(target.reshape(1, -1))[0].round(3)
        target_prediction_all = init_explainers.black_box_predict_proba(counterfactual.reshape(1, -1))[0].round(3)
        initial_prediction = initial_prediction_all[init_explainers.target_class]
        target_prediction = target_prediction_all[init_explainers.target_class]
        nb_not_zero = np.count_nonzero(counterfactual_explanation)
        decreasing_precision = [{"initial prediction": initial_prediction, "counterfactual prediction": target_prediction, \
            "target class":init_explainers.target_class_name, "counterfactual class":init_explainers.counterfactual_class_name}]
        counterfactual_explanation = counterfactual_explanation.tolist()
        if nb_not_zero>1:
            prediction = initial_prediction + 0.0001
            for i in range(nb_not_zero):
                for nb, feature_value in enumerate(counterfactual_explanation):
                    if feature_value != 0:
                        target_temp = target.copy()
                        target_temp[nb] -= feature_value
                        modified_prediction = init_explainers.black_box_predict_proba(target_temp.reshape(1, -1))[0][init_explainers.target_class].round(3)
                        if modified_prediction < prediction:
                            prediction = modified_prediction
                            feature = nb
                to_add = {'prediction': prediction, 'feature name': init_explainers.feature_names[feature]}
                if feature in init_explainers.categorical_features:
                    to_add['value cf'] = init_explainers.feature_transformations[feature](counterfactual[feature])
                    to_add['value target'] = init_explainers.feature_transformations[feature](target_instance[feature])
                else:
                    to_add['value cf'] = str(round(counterfactual[feature], 2))
                    to_add['value target'] = str(target_instance[feature])
                decreasing_precision.append(to_add)
                target[feature] -= counterfactual_explanation[feature]
                counterfactual_explanation[feature] = 0 
            save_dictionary_to_csv(init_explainers.filename + "counterfactual.csv", decreasing_precision)
            return decreasing_precision
        else:
            feature = [i for i in range(len(counterfactual_explanation)) if counterfactual_explanation[i] != 0][0]
            feature = int(feature)
            decreasing_precision.append({'prediction':  target_prediction, \
                'feature name': init_explainers.feature_names[feature], \
                    'value cf': init_explainers.feature_transformations[feature](counterfactual[feature]), \
                        'value target': init_explainers.feature_transformations[feature](target_instance[feature])})
            save_dictionary_to_csv(init_explainers.filename + "counterfactual.csv", decreasing_precision)
            return decreasing_precision