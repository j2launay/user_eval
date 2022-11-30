import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as transforms
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
    categorical = False if ("Age" in feature or "age" in feature or "Height" in feature) else True
    rule = feature.split("<=")
    if len(rule) == 2:
        temp_rule = rule[0].split("<")
        if len(temp_rule) == 2:
            rule[0] = temp_rule[1][1:]
            rule[1] = "between " + str(int(float(temp_rule[0]))) + " and " + str(int(float(rule[1])))
        else:
            rule[1] = "<= " + str(int(float(rule[1]))) if not categorical else rule[1]
    if len(rule) == 1:
        rule = rule[0].split(">=")
        if len(rule) == 2:
            temp_rule = rule[0].split(">")
            if len(temp_rule) == 2:
                rule[0] = temp_rule[1][1:]
                rule[1] = "between " + rule[0] + " and " + temp_rule[1]
            else:
                rule[1] = ">= " + str(int(float(rule[1]))) if not categorical else rule[1]
    if len(rule) == 1:
        rule = rule[0].split("=")
        if len(rule) == 2:
            rule[1] = "= " + str(int(float(rule[1]))) if not categorical else rule[1]
    if len(rule) == 1:
        rule = rule[0].split("<")
        if len(rule) == 2:
            rule[1] = "< " + str(int(float(rule[1]))) if not categorical else rule[1]
    if len(rule) == 1:
        rule = rule[0].split(">")
        if len(rule) == 2:
            rule[1] = "> " + str(int(float(rule[1]))) if not categorical else rule[1]
    return rule

def anchor_image_fancy(init_explainers, ax, filename, height_legend, n_cols):
    #plt.title("anchor rule for " + r"${\bf " + add_space(init_explainers.target_class_name) + r"}$", fontsize=20, y=1.1)
    plt.ylim([0, 100])
    plt.xlim([0, 100])
    ax.set_yticks([0.5], labels="")
    ax.set_xticks([20, 40, 60, 80, 100], labels=["20%", "40%", "60%", "80%", "100%"], size=16)
    plt.xlabel("Confidence", fontsize=16)
    #ax.yaxis.set_tick_params(width=10)
    plt.subplots_adjust(bottom=0.1, top=0.75)
    #ax.set_xticks(np.arange(0, 101, 10), labels=[str(x) for x in np.arange(0, 101, 10)], size=16)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, height_legend),
          fancybox=True, shadow=True, fontsize=14, ncol=n_cols)
    plt.savefig(filename + "rule_image.png")
    plt.show(block=False)
    plt.pause(0.1)
    plt.close('all')

def counterfactual_image_fancy(ax, filename, height_legend, n_cols):
    plt.ylim([0, 1])
    plt.ylabel('')
    ax.set_yticks([0.5], labels="")
    #ax.set_xticks([0.25, 0.5, 0.75], labels=["25%", "50%", "75%"])
    #ax.yaxis.set_tick_params(width=10)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, height_legend),
          fancybox=True, shadow=True, fontsize=14, ncol=n_cols)
    plt.subplots_adjust(bottom=0.1, top=0.8)
    plt.savefig(filename + "counterfactual_image.png")
    plt.show(block=False)
    plt.pause(0.1)
    plt.close('all')

def linear_image_fancy(ax, init_explainers, y_lim, height):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.xaxis.set_tick_params(width=0)
    # TODO change height
    left, bottom, width, height = (0, -0.5, 0.25, height)#5)
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
    # Generate the label of each class prediction
    """ax.set_xticks([0, 0.12, 0.37, 0.62, 0.87, 1], labels=[0, transform_target_class(0.12, init_explainers.class_names), 
                transform_target_class(0.37, init_explainers.class_names), transform_target_class(0.62, init_explainers.class_names), 
                    transform_target_class(0.87, init_explainers.class_names), 100], size=20)"""
    ax.set_xticks([0, 1], labels=[" ", " "], size=20)
    plt.text(0.04, y_lim, transform_target_class(0.12, init_explainers.class_names), fontsize=20)
    plt.text(0.33, y_lim, transform_target_class(0.37, init_explainers.class_names), fontsize=20)
    plt.text(0.55, y_lim, transform_target_class(0.62, init_explainers.class_names), fontsize=20)
    plt.text(0.82, y_lim, transform_target_class(0.87, init_explainers.class_names), fontsize=20)

def linear_image_fancy_end(ax, filename, right_array_graph_value):
    ax.yaxis.set_tick_params(width=0, pad=20)
    ax.margins(1, 0)
    plt.subplots_adjust(left=0.32)
    """for nb, right_value in enumerate(right_array_graph_value):
        if right_value[1] == "R":
            arrow = mpatches.FancyArrow(right_value[0], len(right_array_graph_value)-nb-1, 0.0001, 0, head_width=0.75, head_length=0.03, color="Red")
        else:
            arrow = mpatches.FancyArrow(right_value[0], len(right_array_graph_value)-nb-1, -0.0001, 0, head_width=0.75, head_length=0.03, color="Blue")
        ax.add_line(arrow)"""
    plt.savefig(filename + "linear_image.png")
    plt.show(block=False)
    plt.pause(0.1)
    plt.close('all')

def add_space(text):
    if "<" in text or "=" in text or ">" in text:
        return text
    text_tab = text.split(' ')
    for i, text in enumerate(text_tab):
        text_tab[i] = text + "\ "
    text = ' '.join(text_tab)
    return text

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
        fig, ax = plt.subplots(figsize=(18, 7))
        anchor_rule = {}
        last_precision = 0
        for rule, precision in zip(anchor_exp.names(), anchor_exp.exp_map['precision']):
            init_precision = precision
            anchor_rule[rule] = precision
            rule = split_explanation(rule)
            rule[0], nb_feature = modify_feature_name[rule[0][:-1]][0], modify_feature_name[rule[0][:-1]][1]
            if nb_feature in categorical_features:
                try:
                    answer = feature_transformations[nb_feature](rule[1])
                except KeyError:
                    answer = rule[1]
            else:
                answer = rule[1]
            precision = int(precision*100)
            label = rule[0] + " " + r"${\bf " + add_space(answer) + r"}$"
            plt.barh([50], precision-last_precision, height=60, label=label, left=last_precision)
            last_precision = precision
            
            mytrans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
            ax.arrow(last_precision,0.9,0,0.1,transform=mytrans, head_width=2, head_length=0.1, length_includes_head=True, clip_on=False, color="black")
            plt.axvline(last_precision, ymax=0.9, color="black", linewidth=4)

            ax.annotate(text = str(last_precision) + "%", xy =(init_precision-0.02, 1.01), xycoords='axes fraction', color="black", size=14)
        save_dictionary_to_csv(filename_per_instance + "anchor_rule.csv", [anchor_rule])
        n_cols = int((len(anchor_exp.names()) + 1) / 2)
        anchor_image_fancy(init_explainers, ax, filename_per_instance, 1.1 + ((len(anchor_exp.names())/n_cols)/10), n_cols=n_cols)
        return anchor_rule

    def generate_counterfactual_image(self, counterfactual_exp, modify_feature_name, filename_per_instance, init_explainers, initial_prediction_superior):
        last_prediction = counterfactual_exp[0]['initial prediction']        
        target_instance_text = [["If you change:"]]
        fig, ax = plt.subplots(figsize=(18, 7))
        linear_image_fancy(ax, init_explainers, y_lim=1.02, height=1.5)#-0.05)

        # Generate the bar indicating the initial prediction of the AI model as well as the prediction by the model
        mytrans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        ax.arrow(last_prediction,0.1,0,-0.1,transform=mytrans, head_width=0.02, head_length=0.1, 
                length_includes_head=True, color="black", clip_on=False)
        plt.axvline(last_prediction, ymin=0.1, color="black", linewidth=4)
        x_bounds = ax.get_xlim()
        ax.annotate(text = "AI's prediction",#: " +r"${\bf "+ add_space(init_explainers.target_class_name) + r"}$", 
                xy =(min(1, max(-0.1, ((last_prediction-x_bounds[0])/(x_bounds[1]-x_bounds[0])) - 0.08)), -0.07), 
                xycoords='axes fraction', color="black", size=20)
        
        if initial_prediction_superior:
            last_value = last_prediction
        else:
            last_value = None
        for nb, counterfactual in enumerate(counterfactual_exp[1:]):
            feature_translate = modify_feature_name[counterfactual['feature name']][0]
            if initial_prediction_superior:
                label = feature_translate + ' changing from ' + r"${\bf " + add_space(counterfactual['value target']) + r"}$" + 'to ' + \
                        r"${\bf " + add_space(counterfactual['value cf']) + r"}$" + 'reduces prediction by ' + str(int(float((last_prediction - counterfactual['prediction']) * 100))) + "%"
                plt.barh([0.5], last_value-counterfactual['prediction'], height=0.6, left=counterfactual['prediction'], label=label)

            else:
                label = feature_translate + ' changing from ' + r"${\bf " + add_space(counterfactual['value target']) + r"}$" + 'to ' + \
                        r"${\bf " + add_space(counterfactual['value cf']) + r"}$" + 'increases prediction by ' + str(int(float((counterfactual['prediction'] - last_prediction) * 100))) + "%"
                last_value = last_prediction if last_value is None else last_value 
                plt.barh([0.5], counterfactual['prediction'] - last_value, height=0.6, left=last_value, label=label)
            last_value = counterfactual['prediction']
            target_instance_text.append([feature_translate, counterfactual['value target'], "into", counterfactual['value cf']])
            last_prediction = counterfactual['prediction']

        # Generate the bar indicating the final prediction of the AI model as well as the prediction by the model
        plt.axvline(last_prediction, ymin=0.1, color="black", linewidth=4, linestyle='--')
        ax.arrow(last_prediction,0.1,0,-0.1,transform=mytrans, head_width=0.02, head_length=0.1, 
                length_includes_head=True, color="black", clip_on=False)

        x_bounds = ax.get_xlim()
        ax.annotate(text = "Alternative prediction",#: " + r"${\bf " + add_space(init_explainers.counterfactual_class_name) + r"}$", 
                xy =(min(0.9, max(-0.1, ((last_prediction-x_bounds[0])/(x_bounds[1]-x_bounds[0])) - 0.08)), -0.07), 
                xycoords='axes fraction', color="black", size=20)
        
        target_instance_text.append(['the prediction would be ' + init_explainers.counterfactual_class_name])# + " with a probability of " + str(last_prediction)])
        save_dictionary_to_csv(filename_per_instance + "counterfactual_text.csv", target_instance_text, dictionnaire=False)
        n_cols = int((nb + 2)/2)
        counterfactual_image_fancy(ax, filename_per_instance, 1.2 + nb/n_cols/10, n_cols=n_cols)

    def normalize_linear_explanation(self, lime, target_predict_proba):
        sum_coefficient_features = 0
        for element in lime.as_list():
            sum_coefficient_features += abs(element[1])

        normalized_coefficient_values = []
        for element in lime.as_list():
            normalized_coefficient_values.append(element[1]/abs(sum_coefficient_features)*target_predict_proba)

        lime_exp_normalised = []
        nb_feature, old_coef, sum_features_use = 0, 0, 0
        for coefficient, lime_exp in zip(normalized_coefficient_values, lime.as_list()):
            temp_difference_coefficient_value = abs(old_coef) - abs(coefficient)
            if nb_feature > 1 and (nb_feature > 4 or temp_difference_coefficient_value > abs(coefficient)/1.5):
                break
            else:
                old_coef = coefficient
            lime_exp_normalised.append([lime_exp[0], coefficient])
            sum_features_use += coefficient
            nb_feature += 1
        sum_all_features = sum(normalized_coefficient_values)
        sum_features_not_use = sum_all_features - sum_features_use
        return lime_exp_normalised, sum_features_not_use
    
    def generate_linear_image_explanation(self, pos_lime_exp, neg_lime_exp, filename, init_explainer, other_features_sum_values):
        def store_graph_values_left(feature_lime_exp, left_array_graph_value, right_array_graph_value):
            # Store values from last feature to start the graph representation at the last black box prediction
            last = left_array_graph_value[-1]
            for feature_exp in feature_lime_exp:
                last += feature_exp[2]
                left_array_graph_value.append(last)
                if feature_exp[2] > 0:
                    right_array_graph_value.append([last, "R"])
                else:
                    right_array_graph_value.append([last, "L"])
            return left_array_graph_value, right_array_graph_value
        left_array_graph_value = [0.5]
        left_array_graph_value, right_array_graph_value = store_graph_values_left(pos_lime_exp, left_array_graph_value, [])
        left_array_graph_value, right_array_graph_value = store_graph_values_left(neg_lime_exp, left_array_graph_value, right_array_graph_value)
        
        labels = ["Other factors"]
        labels += [x[0] + "\n" + r" ${\bf " + add_space(x[1]) + r"}$" for x in reversed(neg_lime_exp)]
        labels += [x[0] + "\n" + r" ${\bf " + add_space(x[1]) + r"}$" for x in reversed(pos_lime_exp)]

        plt.rcdefaults()
        fig, ax = plt.subplots(figsize=(18, 7))
        linear_image_fancy(ax, init_explainer, y_lim=len(labels)-0.35, height=len(labels))

        label_size = 18
        
        # Generate the bar indicating the final prediction of the AI model as well as the prediction by the model
        plt.axvline(round(left_array_graph_value[-1] + other_features_sum_values, 2), ymin=0.01, color="black", linewidth=4)
        x_bounds = ax.get_xlim()
        mytrans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        ax.arrow(round(left_array_graph_value[-1] + other_features_sum_values, 2),0.01,0,-.05,transform=mytrans, head_width=0.02, head_length=0.05, 
                length_includes_head=True, color="black", clip_on=False)
        ax.annotate(text = "AI's Prediction",#: " + r"${\bf " + init_explainer.target_class_name + r"}$", 
            xy =(max(min((((left_array_graph_value[-1] + \
                other_features_sum_values)-x_bounds[0])/(x_bounds[1]-x_bounds[0])), 0.95) - 0.1, 0), -0.1), 
                    xycoords='axes fraction', color="black", size=label_size+4)
        
        # Color of the other features sum values
        other_color = "Blue" if other_features_sum_values < 0 else "Red"
        ax.barh([0], [round(other_features_sum_values, 2)], left=[left_array_graph_value[-1]], color=other_color)
        # Pos features from lime exp
        ax.barh(np.arange(len(neg_lime_exp) + 1, len(pos_lime_exp) + \
            len(neg_lime_exp) + 1), [x[2] for x in reversed(pos_lime_exp)], \
            left=left_array_graph_value[:len(pos_lime_exp)][::-1], color="Red", alpha=1)
        # Neg features from lime exp
        ax.barh(np.arange(1, len(neg_lime_exp) + 1), [x[2] for x in reversed(neg_lime_exp)], \
                left=left_array_graph_value[len(pos_lime_exp): len(pos_lime_exp) + len(neg_lime_exp)][::-1], 
                    color="Blue", alpha=1)
        # Generate the y labels with question and answers from the user
        ax.set_yticks(np.arange(len(pos_lime_exp) + len(neg_lime_exp) + 1), labels=labels, size=label_size)
        
        # Add the score of each feature value at the side of the bar
        labels_bar = [["+" + str(int(float(other_features_sum_values * 100))) + "%" if other_features_sum_values > 0 \
            else str(int(float(other_features_sum_values * 100))) + "%"],
                ["+" + str(int(float(x[2] * 100))) + "%" for x in reversed(pos_lime_exp)], [str(int(float(x[2] * 100))) + "%" for x in reversed(neg_lime_exp)]]
        for bars, labels in zip(ax.containers, labels_bar):
            ax.bar_label(bars, labels=labels, size=label_size+2, padding=6)

        right_array_graph_value += [[right_array_graph_value[-1][0] + other_features_sum_values, "R" if other_features_sum_values > 0 else "L"]]
        linear_image_fancy_end(ax, filename, right_array_graph_value)

    def generate_linear_text_explanation(self, lime, modify_feature_name, filename, target_instance, 
                        categorical_features, feature_transformations, other_features_sum_values, init_explainer):
        final_text = []
        pos_graphical_answer_question, neg_graphical_answer_question = [], []
        for lime_exp in lime:
            feature = lime_exp[0]
            rule = split_explanation(feature)
            question, nb_feature = modify_feature_name[rule[0]][0], modify_feature_name[rule[0]][1]
            if nb_feature in categorical_features:
                try:
                    answer = str(feature_transformations[nb_feature](rule[1]))
                except KeyError:
                    answer = rule[1]
            else:
                answer = str(target_instance[nb_feature])

            if lime_exp[1] > 0:
                pos_graphical_answer_question.append([question, answer, lime_exp[1]])
            else:
                neg_graphical_answer_question.append([question, answer, lime_exp[1]])
            
            text = "answering " + answer + " to " + question
            text += " increases " if lime_exp[1] > 0 else " reduces "
            text += "the chance of being \n " + init_explainer.target_class_name + " \n by " + str(int(float(lime_exp[1] * 100)))
            final_text.append([text])

        additional_text = "the other factors increases " if other_features_sum_values > 0 else "the other factors reduces " 
        additional_text += "the prediction of " + str(int(float(other_features_sum_values * 100))) + " to be " + init_explainer.target_class_name
        final_text.append([additional_text])
        save_dictionary_to_csv(filename + "linear_text.csv", final_text, dictionnaire=False)
        return pos_graphical_answer_question, neg_graphical_answer_question, other_features_sum_values

    def generate_counterfactual_text(self, init_explainers, counterfactual, target_instance):
        target = target_instance.copy()
        counterfactual_explanation = target-counterfactual
        initial_prediction_all = init_explainers.black_box_predict_proba(target.reshape(1, -1))[0].round(3)
        target_prediction_all = init_explainers.black_box_predict_proba(counterfactual.reshape(1, -1))[0].round(3)
        initial_prediction = initial_prediction_all[1]#init_explainers.target_class]
        target_prediction = target_prediction_all[1]#init_explainers.target_class]
        initial_prediction_superior = True if initial_prediction > target_prediction else False
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
                        modified_prediction = init_explainers.black_box_predict_proba(target_temp.reshape(1, -1))[0][1].round(3)#init_explainers.target_class].round(3)
                        if modified_prediction < prediction and initial_prediction_superior or (modified_prediction > prediction and not initial_prediction_superior):
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
            return decreasing_precision, initial_prediction_superior
        else:
            feature = [i for i in range(len(counterfactual_explanation)) if counterfactual_explanation[i] != 0][0]
            feature = int(feature)
            value_cf = init_explainers.feature_transformations[feature](counterfactual[feature]) if feature in init_explainers.categorical_features else str(round(counterfactual[feature], 2))
            value_target = init_explainers.feature_transformations[feature](target_instance[feature]) if feature in init_explainers.categorical_features else str(target_instance[feature])
            decreasing_precision.append({'prediction':  target_prediction, \
                'feature name': init_explainers.feature_names[feature], 'value cf': value_cf, \
                    'value target': value_target})
            save_dictionary_to_csv(init_explainers.filename + "counterfactual.csv", decreasing_precision)
            return decreasing_precision, initial_prediction_superior