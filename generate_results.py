import pandas as pd
import numpy as np
import pingouin as pg
from scipy import stats
import scikit_posthocs as sp
import matplotlib.pyplot as plt
import seaborn
import statsmodels.api as sm
from statsmodels.formula.api import ols

pd.options.mode.chained_assignment = None  # default='warn'

def transform_text_to_int(text):
    if text == "Strongly disagree":
        return 1
    elif text == "Somewhat disagree":
        return 2
    elif text == "Neither agree nor disagree":
        return 3
    elif text == "Somewhat agree":
        return 4
    else:
        return 5

def convert_prediction(text, dataset_name):
    if dataset_name == "Obesity":
        if text == "Underweight":
            return 0
        elif text == "Healthy":
            return 1
        elif text == "Overweight":
            return 2
        else:
            return 3
    else:
        if text == "No risk":
            return 0
        elif text == "Low risk":
            return 1
        elif text == "Medium risk":
            return 2
        else:
            return 3


def ensure_valid_answer(data, dataset_name):
    valid_data = data.copy()
    first_question, second_question = valid_data['Q2.2'], valid_data['Q2.4']
    if dataset_name == "Obesity":
        idx_first = np.where(first_question == "Based on weight and height")
        valid_data = valid_data.loc[idx_first]
        idx = np.where((second_question == "Comparing an individual's information with prior observations.") | \
            (second_question == "Calculating the average risk of the entire dataset."))
        idx_first = np.array(idx_first).tolist()[0]
        idx = [x for x in idx[0] if x in idx_first]
        valid_data = valid_data.loc[idx] 
    else:
        idx_first = np.where(first_question == "Help the judge decide whether to release a prisoner.")#
        valid_data = valid_data.loc[idx_first]
        idx = np.where(second_question == "Comparing a prisoner's information with prior observations.")
        idx_first = np.array(idx_first).tolist()[0]
        idx = [x for x in idx[0] if x in idx_first]
        valid_data = valid_data.loc[idx] 

    #valid_data.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
    valid_data.dropna(subset=['Q2.2', 'Q2.4'], inplace=True)
    return valid_data

def add_demographic_information(data, dataset_name, local_surrogates, representations):
    data.rename(columns={"PROLIFIC_PID": "Participant id"}, inplace=True)
    try:
        df1 = pd.read_csv("./user_study/" + dataset_name + "_control.csv")
    except FileNotFoundError:
        pass
    #data = pd.merge(data, df, on = "Participant id")
    for local_surrogate in local_surrogates:
        for representation in representations:
            try:
                df2 = pd.read_csv("./user_study/" + dataset_name + "_" + local_surrogate + "_" + representation + ".csv")
                df1 = pd.concat([df1, df2])
            except FileNotFoundError:
                continue
    keep = data.shape[0]
    data = pd.merge(data, df1, on="Participant id")
    print("duplicate")
    print(data[data['Participant id'].duplicated(keep=False)]['Participant id'])
    if dataset_name != "Compas":
        data = data.drop_duplicates(subset='Participant id', keep="last")
    data.to_csv("./test.csv")
    print(data.iloc[keep:])
    data = data.iloc[:keep]
    return data

def compute_trust(data):
    trust = data[['Duration (in seconds)', 'QT.1', 'QT.2', 'QT.3', 'QT.4', 'local_surrogate', 'representation', 'Sex', 'Age', 'Nationality', 'Highest education level completed', 'Ethnicity simplified']]
    trust['QT.1'].replace({'1. I do not trust it at all.': 1, "7. I trust it completely.":7}, inplace=True)
    trust['QT.2'].replace({'1. It is not at all predictable.': 1, "7. It is completely predictable.":7}, inplace=True)
    trust['QT.3'].replace({'1. It is not at all reliable.': 1, "7. It is completely reliable.":7}, inplace=True)
    trust['QT.4'].replace({'1. It is not at all efficient.': 1, "7. It is completely efficient.":7}, inplace=True)
    trust[['QT.1', 'QT.2', 'QT.3', 'QT.4']] = trust[['QT.1', 'QT.2', 'QT.3', 'QT.4']].astype('int')
    trust['trust'] = trust[['QT.1', 'QT.2', 'QT.3', 'QT.4']].mean(numeric_only=True, axis=1)
    trust[['Duration (in seconds)']].astype('float64')
    return trust[['local_surrogate', 'representation', 'Sex', 'Age', 'Nationality', \
        'Highest education level completed', 'Ethnicity simplified', 'QT.1', 'QT.2', \
            'QT.3', 'QT.4', 'trust', 'Duration (in seconds)']]
    
def compute_satisfaction(data):
    satisfaction = data[['QS.1', 'QS.2', 'QS.3', 'QS.4', 'QS.5', 'QS.6', 'QS.7', 'QS.8', 'local_surrogate', 'representation']]
    for column_name in ['QS.1', 'QS.2', 'QS.3', 'QS.4', 'QS.5', 'QS.6', 'QS.7', 'QS.8']:
        satisfaction[column_name].replace({'I disagree strongly': 1, 'I disagree somewhat': 2, "I'm neutral about it": 3,
                "I agree somewhat": 4, "I agree strongly": 5}, inplace=True)
    satisfaction[['QS.1', 'QS.2', 'QS.3', 'QS.4', 'QS.5', 'QS.6', 'QS.7', 'QS.8']] = satisfaction[['QS.1', 'QS.2', 'QS.3', 'QS.4', 'QS.5', 'QS.6', 'QS.7', 'QS.8']].astype('int')
    satisfaction['satisfaction'] = satisfaction[['QS.1', 'QS.2', 'QS.3', 'QS.4', 'QS.5', 'QS.6', 'QS.7', 'QS.8']].mean(numeric_only=True, axis=1)
    return satisfaction[['QS.1', 'QS.2', 'QS.3', 'QS.4', 'QS.5', 'QS.6', 'QS.7', 'QS.8', 'satisfaction']]#[['satisfaction', 'local_surrogate', 'representation']]
    #print(satisfaction)

def compute_understanding(data):
    understanding = data[['QU.1', 'QU.2', 'QU.3', 'QU.4', 'QU.5', 'QU.6', 'QU.7', 'QU.8', 'local_surrogate', 'representation']]
    for column_name in ['QU.1', 'QU.2', 'QU.3', 'QU.4', 'QU.5', 'QU.6', 'QU.7', 'QU.8']:
        understanding[column_name].replace({'I disagree strongly': 1, 'I disagree somewhat': 2, "I'm neutral about it": 3,
                "I agree somewhat": 4, "I agree strongly": 5}, inplace=True)
    understanding[['QU.1', 'QU.2', 'QU.3', 'QU.4', 'QU.5', 'QU.6', 'QU.7', 'QU.8']] = understanding[['QU.1', 'QU.2', 'QU.3', 'QU.4', 'QU.5', 'QU.6', 'QU.7', 'QU.8']].astype('int')
    understanding['understanding'] = understanding[['QU.1', 'QU.2', 'QU.3', 'QU.4', 'QU.5', 'QU.6', 'QU.7', 'QU.8']].mean(numeric_only=True, axis=1)
    return understanding[['QU.1', 'QU.2', 'QU.3', 'QU.4', 'QU.5', 'QU.6', 'QU.7', 'QU.8', 'understanding']]#[['understanding', 'local_surrogate', 'representation']]
    #print(understanding)

def compute_behavioral_understanding(data, dataset_name):
    # TODO evaluer pour Control group
    if dataset_name == "Obesity":
        linear_q1_solution = ['Family member has overweight', 'Consumption of food between meals', 'Frequent consumption of high caloric food', 'Transportation used', 'Calories consumption monitoring']
        linear_q2_solution = ['Family member has overweight', 'Frequent consumption of high caloric food']
        linear_q3_solution = ['Family member has overweight', 'Consumption of food between meals', 'Frequent consumption of high caloric food', 'Frequency of consumption of vegetables', 'Calories consumption monitoring']
        linear_q4_solution = ['Consumption of food between meals', 'Frequent consumption of high caloric food', 'Age', 'Calories consumption monitoring']
        rule_q1_solution = ['Age', 'Gender', 'Transportation used', 'Consumption of alcohol', 'Physical activity frequency']
        rule_q2_solution = ['Age', 'Family member has overweight', 'Consumption of alcohol', 'Frequent consumption of high caloric food']
        rule_q3_solution = ['Age', 'Family member has overweight', 'Physical activity frequency']
        rule_q4_solution = ['Age', 'Height', 'Consumption of daily water', 'Frequent consumption of high caloric food']
        cf_q1_solution = ['Calories consumption monitoring']
        cf_q2_solution = ['Physical activity frequency']
        cf_q3_solution = ['Family member has overweight']
        cf_q4_solution = ['Family member has overweight', "Physical activity frequency"]
        #control_solution_q1 = ['Family member has overweight', 'Consumption of food between meals', 'Frequent consumption of high caloric food', 'Transportation used', 'Calories consumption monitoring', 'Age', 'Gender', 'Consumption of alcohol', 'Physical activity frequency' ]
        #control_solution_q2 = ['Family member has overweight', 'Frequent consumption of high caloric food', 'Age', 'Consumption of alcohol', 'Physical activity frequency']
        #control_solution_q3 = ['Family member has overweight', 'Consumption of food between meals', 'Frequent consumption of high caloric food', 'Frequency of consumption of vegetables', 'Calories consumption monitoring', 'Age', 'Physical activity frequency']
        #control_solution_q4 = ['Consumption of food between meals', 'Frequent consumption of high caloric food', 'Age', 'Calories consumption monitoring', 'Age', 'Height', 'Consumption of daily water', "Physical activity frequency"]
        linear_q1_top_solution = ['Family member has overweight']
        linear_q2_top_solution = ['Family member has overweight']
        linear_q3_top_solution = ['Family member has overweight']
        linear_q4_top_solution = ['Consumption of food between meals']#Family member has overweight
        rule_q1_top_solution = ['Age']
        rule_q2_top_solution = ['Age']
        rule_q3_top_solution = ['Family member has overweight']
        rule_q4_top_solution = ['Frequent consumption of high caloric food']
        cf_q1_top_solution = ['Calories consumption monitoring']
        cf_q2_top_solution = ['Physical activity frequency']
        cf_q3_top_solution = ['Family member has overweight']
        cf_q4_top_solution = ['Family member has overweight']
        #control_q1_top_solution = ['Calories consumption monitoring']
        #control_q2_top_solution = ['Family member has overweight']
        #control_q3_top_solution = ['Family member has overweight']
        #control_q4_top_solution = ['Age']
    else:
        linear_q1_solution = ['Number of previous arrest', 'Number of juvenile minor offenses']
        linear_q2_solution = ['Number of juvenile minor offenses', 'Age', 'Number of previous arrest']
        linear_q3_solution = ['Number of previous arrest', 'Age', 'Number of juvenile major offenses']
        linear_q4_solution = ['Number of juvenile minor offenses']
        rule_q1_solution = ['Age', 'Number of previous arrest']
        rule_q2_solution = ['Number of previous arrest', 'Description of the charge', 'Number of juvenile minor offenses']
        rule_q3_solution = ['Race', 'Gender', 'Number of previous arrest', 'The degree of the charge', 'Description of the charge']
        rule_q4_solution = ['Number of previous arrest', 'Number of juvenile major offenses', 'Number of juvenile minor offenses']
        cf_q1_solution = ['Number of previous arrest']
        cf_q2_solution = ['Number of previous arrest', 'The degree of the charge']
        cf_q3_solution = ['Number of previous arrest', 'The degree of the charge']
        cf_q4_solution = ['Number of previous arrest', 'Number of juvenile major offenses']
        #control_solution_q1 = ['Number of previous arrest', 'Number of juvenile minor offenses', 'Age']
        #control_solution_q2 = ['Number of juvenile minor offenses', 'Age', 'Number of previous arrest', 'The degree of the charge', 'Description of the charge']
        #control_solution_q3 = ['Number of previous arrest', 'The degree of the charge', 'Race', 'Gender', 'Description of the charge']
        #control_solution_q4 = ['Number of juvenile minor offenses', 'Age', 'Number of previous arrest', 'Number of juvenile major offenses']
        linear_q1_top_solution = ['Number of previous arrest']
        linear_q2_top_solution = ['Number of juvenile minor offenses']
        linear_q3_top_solution = ['Number of previous arrest']
        linear_q4_top_solution = ['Number of juvenile minor offenses']
        rule_q1_top_solution = ['Number of previous arrest']
        rule_q2_top_solution = ['Number of previous arrest']
        rule_q3_top_solution = ['Number of previous arrest']
        rule_q4_top_solution = ['Number of previous arrest']
        cf_q1_top_solution = ['Number of previous arrest']
        cf_q2_top_solution = ['Number of previous arrest']
        cf_q3_top_solution = ['Number of previous arrest']
        cf_q4_top_solution = ['Number of previous arrest']
        #control_q1_top_solution = ['']
        #control_q2_top_solution = ['']
        #control_q3_top_solution = ['']
        #control_q4_top_solution = ['']

    linear_group = data.loc[data['local_surrogate'] == 0]
    rule_group = data.loc[data['local_surrogate'] == 1]
    cf_group = data.loc[data['local_surrogate'] == 2]
    control_group = data.loc[data['local_surrogate'] == 3]

    linear_answers_q1 = linear_group['Q6.2']
    linear_answers_q2 = linear_group['Q8.2']
    linear_answers_q3 = linear_group['Q10.2']
    linear_answers_q4 = linear_group['Q12.2']
    rule_answers_q1 = rule_group['Q6.2']
    rule_answers_q2 = rule_group['Q8.2']
    rule_answers_q3 = rule_group['Q10.2']
    rule_answers_q4 = rule_group['Q12.2']
    cf_answers_q1 = cf_group['Q6.2']
    cf_answers_q2 = cf_group['Q8.2']
    cf_answers_q3 = cf_group['Q10.2']
    cf_answers_q4 = cf_group['Q12.2']
    control_answers_q1 = control_group['Q6.2']
    control_answers_q2 = control_group['Q8.2']
    control_answers_q3 = control_group['Q10.2']
    control_answers_q4 = control_group['Q12.2']
    
    def compute_understanding(answers, solutions, control=False):
        precision_users, recall_users = [], []
        if control:
            print("answers")
            print(answers)
            print("solution") 
            print(solutions)
            for i, answer in enumerate(answers):
                print(i, answer)
                temp_recall_users, temp_precision_users = [], []
                for solution in solutions:
                    print(solution)
                    answer_split = answer.split(",")
                    temp_score = 0
                    for answer_temp in answer_split:
                        temp_score += 1 if answer_temp in solution else 0
                    temp_precision_users.append(temp_score/len(answer_split))
                    temp_recall_users.append(temp_score/len(solution))
                precision_users.append(np.mean(temp_precision_users))
                recall_users.append(np.mean(temp_recall_users))
            print("precision", precision_users, solutions)
            #print(answers)
        else:
            for answer in answers:
                answer_split = answer.split(",")
                temp_score = 0
                for answer_temp in answer_split:
                    temp_score += 1 if answer_temp in solutions else 0
                precision_users.append(temp_score/len(answer_split))
                recall_users.append(temp_score/len(solutions))
        return precision_users, recall_users

    def compute_top_understanding(answers, solutions, control=False):
        score = []
        if control:
            for answer in answers:
                temp_score = 0
                for solution in solutions:
                    if solution[0] in answer:
                        temp_score += 1
                score.append(temp_score/len(solutions))
            print("score", score, solutions)
            #print(answers)
        else:
            for answer in answers:
                if solutions[0] in answer:
                    score.append(1)
                else:
                    score.append(0) 
        return score
 
    
    linear_group.loc[:, 'Q1 b precision understanding'], linear_group.loc[:, 'Q1 b recall understanding'] = compute_understanding(linear_answers_q1, linear_q1_solution)
    linear_group.loc[:, 'Q2 b precision understanding'], linear_group.loc[:, 'Q2 b recall understanding'] = compute_understanding(linear_answers_q2, linear_q2_solution)
    linear_group.loc[:, 'Q3 b precision understanding'], linear_group.loc[:, 'Q3 b recall understanding'] = compute_understanding(linear_answers_q3, linear_q3_solution)
    linear_group.loc[:, 'Q4 b precision understanding'], linear_group.loc[:, 'Q4 b recall understanding'] = compute_understanding(linear_answers_q4, linear_q4_solution)
    rule_group.loc[:, 'Q1 b precision understanding'], rule_group.loc[:, 'Q1 b recall understanding'] = compute_understanding(rule_answers_q1, rule_q1_solution)
    rule_group.loc[:, 'Q2 b precision understanding'], rule_group.loc[:, 'Q2 b recall understanding'] = compute_understanding(rule_answers_q2, rule_q2_solution)
    rule_group.loc[:, 'Q3 b precision understanding'], rule_group.loc[:, 'Q3 b recall understanding'] = compute_understanding(rule_answers_q3, rule_q3_solution)
    rule_group.loc[:, 'Q4 b precision understanding'], rule_group.loc[:, 'Q4 b recall understanding'] = compute_understanding(rule_answers_q4, rule_q4_solution)
    cf_group.loc[:, 'Q1 b precision understanding'], cf_group.loc[:, 'Q1 b recall understanding'] = compute_understanding(cf_answers_q1, cf_q1_solution)
    cf_group.loc[:, 'Q2 b precision understanding'], cf_group.loc[:, 'Q2 b recall understanding'] = compute_understanding(cf_answers_q2, cf_q2_solution)
    cf_group.loc[:, 'Q3 b precision understanding'], cf_group.loc[:, 'Q3 b recall understanding'] = compute_understanding(cf_answers_q3, cf_q3_solution)
    cf_group.loc[:, 'Q4 b precision understanding'], cf_group.loc[:, 'Q4 b recall understanding'] = compute_understanding(cf_answers_q4, cf_q4_solution)
    control_group.loc[:, 'Q1 b precision understanding'], control_group.loc[:, 'Q1 b recall understanding'] = compute_understanding(control_answers_q1, [linear_q1_solution, rule_q1_solution, cf_q1_solution], True)
    control_group.loc[:, 'Q2 b precision understanding'], control_group.loc[:, 'Q2 b recall understanding'] = compute_understanding(control_answers_q2, [linear_q2_solution, rule_q2_solution, cf_q2_solution], True)
    control_group.loc[:, 'Q3 b precision understanding'], control_group.loc[:, 'Q3 b recall understanding'] = compute_understanding(control_answers_q3, [linear_q3_solution, rule_q3_solution, cf_q3_solution], True)
    control_group.loc[:, 'Q4 b precision understanding'], control_group.loc[:, 'Q4 b recall understanding'] = compute_understanding(control_answers_q4, [linear_q4_solution, rule_q4_solution, cf_q4_solution], True)

    q1 = pd.concat([linear_group['Q1 b precision understanding'], rule_group['Q1 b precision understanding'], cf_group['Q1 b precision understanding'], control_group['Q1 b precision understanding']])
    q2 = pd.concat([linear_group['Q2 b precision understanding'], rule_group['Q2 b precision understanding'], cf_group['Q2 b precision understanding'], control_group['Q2 b precision understanding']])
    q3 = pd.concat([linear_group['Q3 b precision understanding'], rule_group['Q3 b precision understanding'], cf_group['Q3 b precision understanding'], control_group['Q3 b precision understanding']])
    q4 = pd.concat([linear_group['Q4 b precision understanding'], rule_group['Q4 b precision understanding'], cf_group['Q4 b precision understanding'], control_group['Q4 b precision understanding']])
    data['Q1 b precision understanding'], data['Q2 b precision understanding'], data['Q3 b precision understanding'], data['Q4 b precision understanding'] = q1, q2, q3, q4
    q1 = pd.concat([linear_group['Q1 b recall understanding'], rule_group['Q1 b recall understanding'], cf_group['Q1 b recall understanding'], control_group['Q1 b recall understanding']])
    q2 = pd.concat([linear_group['Q2 b recall understanding'], rule_group['Q2 b recall understanding'], cf_group['Q2 b recall understanding'], control_group['Q2 b recall understanding']])
    q3 = pd.concat([linear_group['Q3 b recall understanding'], rule_group['Q3 b recall understanding'], cf_group['Q3 b recall understanding'], control_group['Q3 b recall understanding']])
    q4 = pd.concat([linear_group['Q4 b recall understanding'], rule_group['Q4 b recall understanding'], cf_group['Q4 b recall understanding'], control_group['Q4 b recall understanding']])
    data['Q1 b recall understanding'], data['Q2 b recall understanding'], data['Q3 b recall understanding'], data['Q4 b recall understanding'] = q1, q2, q3, q4

    linear_group.loc[:, 'Q1 b top understanding'] = compute_top_understanding(linear_answers_q1, linear_q1_top_solution)
    linear_group.loc[:, 'Q2 b top understanding'] = compute_top_understanding(linear_answers_q2, linear_q2_top_solution)
    linear_group.loc[:, 'Q3 b top understanding'] = compute_top_understanding(linear_answers_q3, linear_q3_top_solution)
    linear_group.loc[:, 'Q4 b top understanding'] = compute_top_understanding(linear_answers_q4, linear_q4_top_solution)
    rule_group.loc[:, 'Q1 b top understanding'] = compute_top_understanding(rule_answers_q1, rule_q1_top_solution)
    rule_group.loc[:, 'Q2 b top understanding'] = compute_top_understanding(rule_answers_q2, rule_q2_top_solution)
    rule_group.loc[:, 'Q3 b top understanding'] = compute_top_understanding(rule_answers_q3, rule_q3_top_solution)
    rule_group.loc[:, 'Q4 b top understanding'] = compute_top_understanding(rule_answers_q4, rule_q4_top_solution)
    cf_group.loc[:, 'Q1 b top understanding'] = compute_top_understanding(cf_answers_q1, cf_q1_top_solution)
    cf_group.loc[:, 'Q2 b top understanding'] = compute_top_understanding(cf_answers_q2, cf_q2_top_solution)
    cf_group.loc[:, 'Q3 b top understanding'] = compute_top_understanding(cf_answers_q3, cf_q3_top_solution)
    cf_group.loc[:, 'Q4 b top understanding'] = compute_top_understanding(cf_answers_q4, cf_q4_top_solution)
    control_group.loc[:, 'Q1 b top understanding'] = compute_top_understanding(control_answers_q1, [linear_q1_top_solution, rule_q1_top_solution, cf_q1_top_solution], True)
    control_group.loc[:, 'Q2 b top understanding'] = compute_top_understanding(control_answers_q2, [linear_q2_top_solution, rule_q2_top_solution, cf_q2_top_solution], True)
    control_group.loc[:, 'Q3 b top understanding'] = compute_top_understanding(control_answers_q3, [linear_q3_top_solution, rule_q3_top_solution, cf_q3_top_solution], True)
    control_group.loc[:, 'Q4 b top understanding'] = compute_top_understanding(control_answers_q4, [linear_q4_top_solution, rule_q4_top_solution, cf_q4_top_solution], True)
    q1 = pd.concat([linear_group['Q1 b top understanding'], rule_group['Q1 b top understanding'], cf_group['Q1 b top understanding'], control_group['Q1 b top understanding']])
    q2 = pd.concat([linear_group['Q2 b top understanding'], rule_group['Q2 b top understanding'], cf_group['Q2 b top understanding'], control_group['Q2 b top understanding']])
    q3 = pd.concat([linear_group['Q3 b top understanding'], rule_group['Q3 b top understanding'], cf_group['Q3 b top understanding'], control_group['Q3 b top understanding']])
    q4 = pd.concat([linear_group['Q4 b top understanding'], rule_group['Q4 b top understanding'], cf_group['Q4 b top understanding'], control_group['Q4 b top understanding']])
    data['Q1 b top understanding'], data['Q2 b top understanding'], data['Q3 b top understanding'], data['Q4 b top understanding'] = q1, q2, q3, q4

    return data[['Q1 b precision understanding', 'Q2 b precision understanding', 'Q3 b precision understanding', 'Q4 b precision understanding']].mean(numeric_only=True, axis=1), \
            data[['Q1 b recall understanding', 'Q2 b recall understanding', 'Q3 b recall understanding', 'Q4 b recall understanding']].mean(numeric_only=True, axis=1), \
                data[['Q1 b top understanding', 'Q2 b top understanding', 'Q3 b top understanding', 'Q4 b top understanding']].mean(numeric_only=True, axis=1)

def compute_behavioral_trust(data, dataset_name):
    if dataset_name == "Obesity":
        first_ai_prediction, second_ai_prediction, third_ai_prediction, fourth_ai_prediction = 'Healthy', 'Overweight', 'Obesity', 'Underweight'
        first_linear_ai_prediction, second_linear_ai_prediction, third_linear_ai_prediction, fourth_linear_ai_prediction = 'Underweight', 'Overweight', 'Obesity', 'Healthy'
    else:
        first_ai_prediction, second_ai_prediction, third_ai_prediction, fourth_ai_prediction = 'High risk', 'No risk', 'Medium risk', 'Low risk'
        first_linear_ai_prediction, second_linear_ai_prediction, third_linear_ai_prediction, fourth_linear_ai_prediction = 'High risk', 'No risk', 'Medium risk', 'Low risk'
    
    def compute_trust(first_prediction, second_prediction, ai_prediction):
        if first_prediction == ai_prediction:
            return None
        else:
            if second_prediction == ai_prediction:
                return 1
            else:
                return 0

    def compute_increased_trust(first_prediction, second_prediction, ai_prediction):
        first_prediction_num = convert_prediction(first_prediction, dataset_name)
        second_prediction_num = convert_prediction(second_prediction, dataset_name)
        ai_prediction_num = convert_prediction(ai_prediction, dataset_name)
        first = abs(ai_prediction_num - first_prediction_num)
        second = abs(ai_prediction_num - second_prediction_num)
        return first - second

    q1, q2, q3, q4 = [], [], [], []
    q1_ordinal, q2_ordinal, q3_ordinal, q4_ordinal = [], [], [], []
    for index, row in data.iterrows():
        first_prediction, first_prediction_after_xai = row['Q5.2'], row['Q6.5']
        second_prediction, second_prediction_after_xai = row['Q7.2'], row['Q8.5']
        third_prediction, third_prediction_after_xai = row['Q9.2'], row['Q10.5']
        fourth_prediction, fourth_prediction_after_xai = row['Q11.2'], row['Q12.5']
        if row['local_surrogate'] == 0:
            q1.append(compute_trust(first_prediction, first_prediction_after_xai, first_linear_ai_prediction))
            q2.append(compute_trust(second_prediction, second_prediction_after_xai, second_linear_ai_prediction))
            q3.append(compute_trust(third_prediction, third_prediction_after_xai, third_linear_ai_prediction))
            q4.append(compute_trust(fourth_prediction, fourth_prediction_after_xai, fourth_linear_ai_prediction))

            q1_ordinal.append(compute_increased_trust(first_prediction, first_prediction_after_xai, first_linear_ai_prediction))
            q2_ordinal.append(compute_increased_trust(second_prediction, second_prediction_after_xai, second_linear_ai_prediction))
            q3_ordinal.append(compute_increased_trust(third_prediction, third_prediction_after_xai, third_linear_ai_prediction))
            q4_ordinal.append(compute_increased_trust(fourth_prediction, fourth_prediction_after_xai, fourth_linear_ai_prediction))
        else:
            q1.append(compute_trust(first_prediction, first_prediction_after_xai, first_ai_prediction))
            q2.append(compute_trust(second_prediction, second_prediction_after_xai, second_ai_prediction))
            q3.append(compute_trust(third_prediction, third_prediction_after_xai, third_ai_prediction))
            q4.append(compute_trust(fourth_prediction, fourth_prediction_after_xai, fourth_ai_prediction))

            q1_ordinal.append(compute_increased_trust(first_prediction, first_prediction_after_xai, first_ai_prediction))
            q2_ordinal.append(compute_increased_trust(second_prediction, second_prediction_after_xai, second_ai_prediction))
            q3_ordinal.append(compute_increased_trust(third_prediction, third_prediction_after_xai, third_ai_prediction))
            q4_ordinal.append(compute_increased_trust(fourth_prediction, fourth_prediction_after_xai, fourth_ai_prediction))

    data['Q1 b trust'], data['Q2 b trust'], data['Q3 b trust'], data['Q4 b trust'] = q1, q2, q3, q4
    data['Q1 b ord trust'], data['Q2 b ord trust'], data['Q3 b ord trust'], data['Q4 b ord trust'] = q1_ordinal, q2_ordinal, q3_ordinal, q4_ordinal
    return data[['Q1 b trust', 'Q2 b trust', 'Q3 b trust', 'Q4 b trust']].mean(numeric_only=True, axis=1),\
            data[['Q1 b ord trust', 'Q2 b ord trust', 'Q3 b ord trust', 'Q4 b ord trust']].mean(numeric_only=True, axis=1)

def compute_increased_trust(data):
    q1, q2, q3, q4 = [], [], [], []
    def compute_increased_trust_ind(first_confidence, first_confidence_after_xai):
        first, second = transform_text_to_int(first_confidence), transform_text_to_int(first_confidence_after_xai)
        return second - first

    for index, row in data.iterrows():
        first_confidence, first_confidence_after_xai = row['Q5.3'], row['Q6.6']
        second_confidence, second_confidence_after_xai = row['Q7.3'], row['Q8.6']
        third_confidence, third_confidence_after_xai = row['Q9.3'], row['Q10.6']
        fourth_confidence, fourth_confidence_after_xai = row['Q11.3'], row['Q12.6']
        q1.append(compute_increased_trust_ind(first_confidence, first_confidence_after_xai))
        q2.append(compute_increased_trust_ind(second_confidence, second_confidence_after_xai))
        q3.append(compute_increased_trust_ind(third_confidence, third_confidence_after_xai))
        q4.append(compute_increased_trust_ind(fourth_confidence, fourth_confidence_after_xai))
    data['Q1 increased trust after xai'], data['Q2 increased trust after xai'], data['Q3 increased trust after xai'], data['Q4 increased trust after xai'] = q1, q2, q3, q4
    return data[['Q1 increased trust after xai', 'Q2 increased trust after xai', 'Q3 increased trust after xai', 'Q4 increased trust after xai']].mean(numeric_only=True, axis=1)

def compute_perceived_trust_in_xai(data):
    q1, q2, q3, q4 = [], [], [], []
    for index, row in data.iterrows():
        first, second, third, fourth = row['Q6.4'], row['Q8.4'], row['Q10.4'], row['Q12.4']
        q1.append(transform_text_to_int(first)) 
        q2.append(transform_text_to_int(second)) 
        q3.append(transform_text_to_int(third)) 
        q4.append(transform_text_to_int(fourth))
    data['Q1 perceived trust'], data['Q2 perceived trust'], data['Q3 perceived trust'], data['Q4 perceived trust'] = q1, q2, q3, q4
    return data[['Q1 perceived trust', 'Q2 perceived trust', 'Q3 perceived trust', 'Q4 perceived trust']].mean(numeric_only=True, axis=1)

def compute_perceived_understanding_in_xai(data):
    q1, q2, q3, q4 = [], [], [], []
    for index, row in data.iterrows():
        first, second, third, fourth = row['Q6.3'], row['Q8.3'], row['Q10.3'], row['Q12.3']
        q1.append(transform_text_to_int(first)) 
        q2.append(transform_text_to_int(second)) 
        q3.append(transform_text_to_int(third)) 
        q4.append(transform_text_to_int(fourth))
    data['Q1 perceived understanding'], data['Q2 perceived understanding'], data['Q3 perceived understanding'], data['Q4 perceived understanding'] = q1, q2, q3, q4
    return data[['Q1 perceived understanding', 'Q2 perceived understanding', 'Q3 perceived understanding', 'Q4 perceived understanding']].mean(numeric_only=True, axis=1)

def print_demographic_latex_table(result_df):
    result_df.rename(columns={'Sex': 'Gender', 'Highest education level completed': 'Highest education'}, inplace=True)
    result_df.loc[result_df['Nationality'] == "South Africa", "Nationality"] = "Africa"
    result_df.loc[result_df['Nationality'] == "Zimbabwe", "Nationality"] = "Africa"
    result_df.loc[result_df['Nationality'] == "Nigeria", "Nationality"] = "Africa"
    result_df.loc[result_df['Nationality'] == "Portugal", "Nationality"] = "Europe"
    result_df.loc[result_df['Nationality'] == "United Kingdom", "Nationality"] = "Europe"
    result_df.loc[result_df['Nationality'] == "Italy", "Nationality"] = "Europe"
    result_df.loc[result_df['Nationality'] == "Greece", "Nationality"] = "Europe"
    result_df.loc[result_df['Nationality'] == "Spain", "Nationality"] = "Europe"
    result_df.loc[result_df['Nationality'] == "Netherlands", "Nationality"] = "Europe"
    result_df.loc[result_df['Nationality'] == "Finland", "Nationality"] = "Europe"
    result_df.loc[result_df['Nationality'] == "France", "Nationality"] = "Europe"
    result_df.loc[result_df['Nationality'] == "Germany", "Nationality"] = "Europe"
    result_df.loc[result_df['Nationality'] == "Norway", "Nationality"] = "Europe"
    result_df.loc[result_df['Nationality'] == "Poland", "Nationality"] = "Europe"
    result_df.loc[result_df['Nationality'] == "Lithuania", "Nationality"] = "Europe"
    result_df.loc[result_df['Nationality'] == "Slovenia", "Nationality"] = "Europe"
    result_df.loc[result_df['Nationality'] == "Hungary", "Nationality"] = "Europe"
    result_df.loc[result_df['Nationality'] == "Bulgaria", "Nationality"] = "Europe"
    result_df.loc[result_df['Nationality'] == "Czech Republic", "Nationality"] = "Europe"
    result_df.loc[result_df['Nationality'] == "Estonia", "Nationality"] = "Europe"
    result_df.loc[result_df['Nationality'] == "Latvia", "Nationality"] = "Europe"
    result_df.loc[result_df['Nationality'] == "Canada", "Nationality"] = "North America"
    result_df.loc[result_df['Nationality'] == "United States", "Nationality"] = "North America"
    result_df.loc[result_df['Nationality'] == "Mexico", "Nationality"] = "North America"
    result_df.loc[result_df['Nationality'] == "Brazil", "Nationality"] = "South America"
    result_df.loc[result_df['Nationality'] == "Chile", "Nationality"] = "South America"
    result_df.loc[result_df['Nationality'] == "China", "Nationality"] = "Asia"
    result_df.loc[result_df['Nationality'] == "Vietnam", "Nationality"] = "Asia"
    result_df.loc[result_df['Nationality'] == "Israel", "Nationality"] = "Asia"
    demographic_headers = ['Gender', 'Age', 'Nationality', 'Highest education', 'Ethnicity simplified']
    textabular = "\n \\centering \n $$ \n \\begin{aligned} \n & \\begin{array}{lrr} \n"
    texheader = "\\hline \\text{Factor} & \\boldsymbol{N} & \\text{\% sample} \\\\\hline "
    texdata = ""
    for column in demographic_headers:
        target_column = result_df[column]
        texdata += "\\text{" + column + "} & & \\\\\n \\hline " 
        if column == "Age":
            target_column = target_column.replace('CONSENT_REVOKED', 2)
            target_column = target_column.astype('int')
            target_column = pd.cut(target_column,
                        bins=[0, 5, 20, 30, 40, 60], 
                        labels=["CONSENT\_REVOKED", "< 20", "20 < 30", "30 < 40", "40 >"])
        temp_df = pd.concat([target_column.value_counts(), (target_column.value_counts()/target_column.count())*100], axis=1)
        for index, row in temp_df.iterrows():
            if index == "CONSENT_REVOKED":
                index = "CONSENT\_REVOKED"
            row = row.round(2)
            texdata += "\\text{" + str(index) + "} & " + str(int(row.iloc[0])) + " & " + str(row.iloc[1]) + "  \\\\\n"
        texdata += "\\hline"
        print(temp_df)
    print("\\begin{table}[ht]"+textabular)
    print(texheader)
    print(texdata)
    print("\\end{array}\\\\\n \\end{aligned} $$ \n \\caption{Overview of participants' demographic factors.} \n \\label{tab:user_information} \n \\end{table}")

def generate_precision_recall_curve(result_df):
    fig, ax = plt.subplots()
    picture_linear = result_df.loc[(result_df['representation'] == 0) & (result_df['local_surrogate'] == 0)]
    picture_rule = result_df.loc[(result_df['representation'] == 0) & (result_df['local_surrogate'] == 1)]
    picture_counterfactual = result_df.loc[(result_df['representation'] == 0) & (result_df['local_surrogate'] == 2)]
    text_linear = result_df.loc[(result_df['representation'] == 1) & (result_df['local_surrogate'] == 0)]
    text_rule = result_df.loc[(result_df['representation'] == 1) & (result_df['local_surrogate'] == 1)]
    text_counterfactual = result_df.loc[(result_df['representation'] == 1) & (result_df['local_surrogate'] == 2)]
    control_group = result_df.loc[(result_df['representation'] == 2) & (result_df['local_surrogate'] == 3)]
    ax.scatter(picture_linear['b_recall_understanding'], picture_linear['b_precision_understanding'], label="linear image")#, color="purple")
    ax.scatter(picture_rule['b_recall_understanding'], picture_rule['b_precision_understanding'], label="rule image")#, color="purple")
    ax.scatter(picture_counterfactual['b_recall_understanding'], picture_counterfactual['b_precision_understanding'], label="cf image")#, color="purple")
    ax.scatter(text_linear['b_recall_understanding'], text_linear['b_precision_understanding'], label="linear text")#, color="purple")
    ax.scatter(text_rule['b_recall_understanding'], text_rule['b_precision_understanding'], label="rule text")#, color="purple")
    ax.scatter(text_counterfactual['b_recall_understanding'], text_counterfactual['b_precision_understanding'], label="cf text")#, color="purple")
    ax.scatter(control_group['b_recall_understanding'], control_group['b_precision_understanding'], label="control")#, color="purple")
    ax.set_title('Precision-Recall Curve')
    ax.set_ylabel('Precision')
    ax.set_xlabel('Recall')
    ax.legend()
    plt.savefig("./user_study/boxplot/precision_recall.png")
    plt.show()
    
def measure_pearson_correlation(result_df):
    print("pearson")
    print(result_df['b_trust'])
    nan_idx = pd.isnull(result_df['b_trust']).to_numpy().nonzero()[0]
    result_without_nan = result_df.drop(index = nan_idx)
    
    print("TRUST")
    print("post questionnaire trust vs perceived trust in xai")
    print(np.corrcoef(result_df['trust'], result_df['perceived trust in xai']))
    print("behavioural trust vs perceived trust in xai")
    print(np.corrcoef(result_without_nan['b_trust'], result_without_nan['perceived trust in xai']))
    print("behavioural ordinal trust vs perceived trust in xai")
    print(np.corrcoef(result_without_nan['b_ord_trust'], result_without_nan['perceived trust in xai']))
    print("behavioural trust vs post questionnaire trust")
    print(np.corrcoef(result_without_nan['b_trust'], result_without_nan['trust']))
    print("post questionnaire trust vs increased trust after xai")
    print(np.corrcoef(result_df['trust'], result_df['increased trust after xai']))
    print()
    
    print("UNDERSTAND")
    print("post questionnaire understand vs perceived understand in xai")
    print(np.corrcoef(result_df['understanding'], result_df['perceived understanding in xai']))
    print("behavioural top understand vs perceived understand in xai")
    print(np.corrcoef(result_df['b_top_understanding'], result_df['perceived understanding in xai']))
    print("behavioural precision understand vs perceived understand in xai")
    print(np.corrcoef(result_df['b_precision_understanding'], result_df['perceived understanding in xai']))
    print("behavioural recall understand vs perceived understand in xai")
    print(np.corrcoef(result_df['b_recall_understanding'], result_df['perceived understanding in xai']))
    print("behavioural top understand vs post questionnaire understand")
    print(np.corrcoef(result_df['b_top_understanding'], result_df['understanding']))
    print("behavioural precision understand vs post questionnaire understand")
    print(np.corrcoef(result_df['b_precision_understanding'], result_df['understanding']))
    print("behavioural recall understand vs post questionnaire understand")
    print(np.corrcoef(result_df['b_recall_understanding'], result_df['understanding']))
    print()
    
    print("SATISFACTION")
    print("post questionnaire satisfaction vs duration")
    print(np.corrcoef(result_df['satisfaction'], result_df['Duration (in seconds)']))
    print("post questionnaire satisfaction vs time")
    print(np.corrcoef(result_df['satisfaction'], result_df['time']))
    
def store_r_dataframe(result_df):
    result_df_r = result_df.rename(columns={'Duration (in seconds)':'Duration_in_seconds', 
                                            'increased trust after xai': 'increased_trust_after_xai',
                                            'perceived trust in xai': "perceived_trust_in_xai", 
                                            'perceived understanding in xai': 'perceived_understanding_in_xai',
                                            'Highest education level completed': 'Highest_education'})
    metrics = ['Highest_education', 'local_surrogate', 'representation']
    
    if dataset_name == "Compas":
        print()
        print([0]*result_df_r.shape[0])
        result_df_r.insert(result_df_r.shape[1], "Highest_education_CONSENT_REVOKED", [0]*result_df_r.shape[0])
        
    for metric in metrics:
        one_hot = pd.get_dummies(result_df_r[metric])
        for i, column in enumerate(one_hot.columns):
            print(i, column)
            one_hot.rename({column: metric + "_" + str(column)}, axis=1, inplace=True)
        # Drop column as it is now encoded
        result_df_r = result_df_r.drop(metric,axis = 1)
        print(result_df_r)
        print(one_hot)
        # Join the encoded df
        result_df_r = result_df_r.join(one_hot)
    result_df_r.to_csv("./user_study/" + dataset_name + "_results_r.csv")
    
if __name__ == "__main__":
    local_surrogates = ['Counterfactual', 'Linear', 'Rule']
    representations = ['picture', 'text']
    dataset_name = "Compas"
    latex, generate_table = True, False
    data = pd.read_csv("./user_study/" + dataset_name + "+Control.csv")
    data['local_surrogate'], data['representation'] = 3, 2#'control', 'control'
    data = ensure_valid_answer(data, dataset_name)
    print("control group", data.shape)
    data = data.iloc[:20]
    for local_surrogate in local_surrogates:
        for representation in representations:
            try:
                temp_df = pd.read_csv("./user_study/" + dataset_name + "+" + local_surrogate + "+" + representation + ".csv")
                if local_surrogate == 'Linear':
                    temp_df['local_surrogate'] = 0 
                elif local_surrogate == "Rule":
                    temp_df['local_surrogate'] = 1
                else:
                    temp_df['local_surrogate'] = 2
                temp_df['representation'] = 0 if representation == "picture" else 1#representation
                temp_df = ensure_valid_answer(temp_df, dataset_name)
                temp_df = temp_df.iloc[:20]
                print(temp_df.shape)
                data = pd.concat([data, temp_df])
            except FileNotFoundError:
                print()
    data.reset_index(inplace=True)
    print("nombre d'individu", len(data))
    data = ensure_valid_answer(data, dataset_name)
    data.reset_index(inplace=True)
    data = add_demographic_information(data, dataset_name, local_surrogates, representations)
    print("now that I remove people", len(data))
    result_df = compute_trust(data)
    print("now that I remove people after trust", len(data))
    result_df[['QU.1', 'QU.2', 'QU.3', 'QU.4', 'QU.5', 'QU.6', 'QU.7', 'QU.8', 'understanding']] = compute_understanding(data)
    print("now that I remove people after understanding", len(data))
    result_df[['QS.1', 'QS.2', 'QS.3', 'QS.4', 'QS.5', 'QS.6', 'QS.7', 'QS.8', 'satisfaction']] = compute_satisfaction(data)
    print("now that I remove people after satisfaction", len(data))
    result_df['b_precision_understanding'], result_df['b_recall_understanding'], result_df['b_top_understanding'] = compute_behavioral_understanding(data, dataset_name)
    result_df['b_trust'], result_df['b_ord_trust'] = compute_behavioral_trust(data, dataset_name)
    result_df['increased trust after xai'] = compute_increased_trust(data)
    result_df['perceived trust in xai'] = compute_perceived_trust_in_xai(data)
    result_df['perceived understanding in xai'] = compute_perceived_understanding_in_xai(data)
    time = data[['Q6.7_Page Submit', 'Q8.7_Page Submit', 'Q10.7_Page Submit', 'Q12.7_Page Submit']].astype('float')
    result_df['time'] = time[['Q6.7_Page Submit', 'Q8.7_Page Submit', 'Q10.7_Page Submit', 'Q12.7_Page Submit']].mean(numeric_only=True, axis=1)
    result_df['Duration (in seconds)'] = result_df['Duration (in seconds)'].astype('float')
    
    predictors = ['local_surrogate', 'representation']
    metrics = ['b_trust', 'understanding', 'trust', 'satisfaction', 'Duration (in seconds)', 'b_precision_understanding', 
               'b_recall_understanding',  'b_top_understanding', "b_ord_trust", 'time', 'increased trust after xai', 
               'perceived trust in xai', 'perceived understanding in xai']#, 'QU.1', 'QU.2', 'QU.3', 'QU.4', 'QU.5',
               #'QU.6', 'QU.7', 'QU.8', 'QS.1', 'QS.2', 'QS.3', 'QS.4', 'QS.5', 'QS.6', 'QS.7', 'QS.8', 'QT.1', 
               #'QT.2', 'QT.3', 'QT.4']
    result_df['domain'] = dataset_name
    result_df.to_csv("./user_study/results.csv")
    
    store_r_dataframe(result_df)
        
    if latex:
        print_demographic_latex_table(result_df)
        result_df['Highest education'].replace({"High school diploma/A-levels": 0, 
                            "Technical/community college": 1, 'Undergraduate degree (BA/BSc/other)': 2, 
                            "Graduate degree (MA/MSc/MPhil/other)": 3, "Doctorate degree (PhD/other)": 4}, inplace=True)
    else:
        result_df['Highest education level completed'].replace({"High school diploma/A-levels": 0, 
                            "Technical/community college": 1, 'Undergraduate degree (BA/BSc/other)': 2, 
                            "Graduate degree (MA/MSc/MPhil/other)": 3, "Doctorate degree (PhD/other)": 4}, inplace=True)
        
    representation_picture = result_df.loc[result_df['representation'] == 0]#.astype('float64')
    representation_text = result_df.loc[result_df['representation'] == 1]#.astype('float64')
    representation_control = result_df.loc[result_df['representation'] == 2]#.astype('float64')
    linear_result = result_df.loc[result_df['local_surrogate'] == 0]#.astype('float64')
    rule_result = result_df.loc[result_df['local_surrogate'] == 1]#.astype('float64')
    counterfactual_result = result_df.loc[result_df['local_surrogate'] == 2]#.astype('float64')

    picture_linear = result_df.loc[(result_df['representation'] == 0) & (result_df['local_surrogate'] == 0)]
    picture_rule = result_df.loc[(result_df['representation'] == 0) & (result_df['local_surrogate'] == 1)]
    picture_counterfactual = result_df.loc[(result_df['representation'] == 0) & (result_df['local_surrogate'] == 2)]
    text_linear = result_df.loc[(result_df['representation'] == 1) & (result_df['local_surrogate'] == 0)]
    text_rule = result_df.loc[(result_df['representation'] == 1) & (result_df['local_surrogate'] == 1)]
    text_counterfactual = result_df.loc[(result_df['representation'] == 1) & (result_df['local_surrogate'] == 2)]
    control_group = result_df.loc[(result_df['representation'] == 2) & (result_df['local_surrogate'] == 3)]
    #generate_precision_recall_curve(result_df)
    
    measure_pearson_correlation(result_df)
    
    for metric in metrics:
        print(metric)

        """print("nb person in linear + image", picture_linear.shape[0])
        print("nb person in linear + text", text_linear.shape[0])
        print("nb person in rule + text", text_rule.shape[0])
        print("nb person in rule + image", picture_rule.shape[0])
        print("nb person in counterfactual + text", text_counterfactual.shape[0])
        print("nb person in counterfactual + image", picture_counterfactual.shape[0])
        print("nb person in control", control_group.shape[0])"""
        
        #perform three-way ANOVA        
        if generate_table:
            print(result_df_r.columns)
            model = ols("""satisfaction ~ C(representation) + C(local_surrogate) + C(Age) +
                    C(representation):C(local_surrogate) + C(representation):C(Age) + 
                    C(Highest_education_level_completed) + C(local_surrogate):C(Age) +
                    C(representation):C(local_surrogate):C(Age)""", data=result_df_r).fit()
            table = sm.stats.anova_lm(model, typ=2)
            print(table)
        
        print("friedman test")
        #print(stats.friedmanchisquare(picture_linear, text_linear, control_group))#,picture_rule, picture_couterfactual, text_rule, text_counterfactual))
        #print("representation")
        #print(stats.friedmanchisquare(picture_linear, control_group))#picture_rule, picture_couterfactual))
        #print("method")
        #print(stats.friedmanchisquare(text_linear, text_rule, text_counterfactual))
        data = np.array([picture_linear[metric], control_group[metric]])#text_linear, picture_rule, picture_couterfactual, text_rule, text_counterfactual, control_group])
        print("Nemenyi post-hoc test")
        print(sp.posthoc_nemenyi_friedman(data.T))

        # Conducting two-sample ttest
        #plt.violinplot([representation_picture, representation_text, representation_control], showmedians=True, inner="points")
        seaborn.swarmplot([representation_picture[metric].tolist(), representation_text[metric].tolist(), 
                           representation_control[metric].tolist()], color= "white")
        seaborn.violinplot([representation_picture[metric], representation_text[metric], 
                            representation_control[metric]], showmedians=True)#, inner="points")
        plt.xticks([0, 1, 2], labels=["image", "text", "control"])
        plt.xlabel("Groups")
        plt.ylabel("measurements")
        plt.title(metric)
        plt.tight_layout
        plt.savefig("./figures/" + dataset_name + "/" + metric.replace(' ', '_') + "_representation.png")
        plt.show()
        seaborn.swarmplot([linear_result[metric].tolist(), rule_result[metric].tolist(), 
                           counterfactual_result[metric].tolist(), control_group[metric].tolist()], color= "white")
        seaborn.violinplot([linear_result[metric].tolist(), rule_result[metric].tolist(), 
                            counterfactual_result[metric].tolist(), control_group[metric].tolist()], showmedians=True)#, inner="points")
        plt.xticks([0, 1, 2, 3], labels=["Linear", "Rules", "Counterfactual", "control"])
        plt.xlabel("Groups")
        plt.ylabel("measurements")
        plt.title(metric)
        plt.tight_layout
        plt.savefig("./figures/" + dataset_name + "/" + metric.replace(' ', '_') + "_surrogate.png")
        plt.show()
  
        nan_idx = pd.isnull(result_df[metric]).to_numpy().nonzero()[0]
        result_without_nan = result_df.drop(index = nan_idx)
        try:
            mod2 = pg.linear_regression(result_without_nan[predictors], result_without_nan[metric])
        except AssertionError as e:
            print()
            print("error")
            print(e)
        #print("multiple linear regression")
        print(mod2.round(3))
        # Print the result
        #print("t-test")
        # T-test for representation
        """
        result = pg.ttest(representation_picture[metric], representation_control[metric])
        print("picture control")
        print(result)
        print()
        result = pg.ttest(representation_text[metric], representation_control[metric])
        print("text control")
        print(result)
        print()        
        result = pg.ttest(representation_picture[metric], representation_text[metric])
        print("picture text")
        print(result)
        print()
          
        # T-test for local surrogate     
        result = pg.ttest(linear_result[metric], rule_result[metric])
        print("linear rule")
        print(result)
        print()
        result = pg.ttest(linear_result[metric], counterfactual_result[metric])
        print("linear counterfactual")
        print(result)
        print()        
        result = pg.ttest(linear_result[metric], control_group[metric])
        print("linear control")
        print(result)
        print()

        result = pg.ttest(counterfactual_result[metric], rule_result[metric])
        print("counterfactual rule")
        print(result)
        print()
        result = pg.ttest(rule_result[metric], control_group[metric])
        print("rule control")
        print(result)
        print()        
        result = pg.ttest(counterfactual_result[metric], control_group[metric])
        print("counterfactual control")
        print(result)
        print()"""
    result_df.to_csv("./user_study/results_" + dataset_name + ".csv")