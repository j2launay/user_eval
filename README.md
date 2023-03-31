# Impact of Explanation Techniques and Visualizations on Users Comprehension and Trust in Explainable AI
This repository contains the code to generate the explanation and visualization to conduct the user experiments for the paper submitted to VIS 2023.
It also includes the code to measure the impact on users of the different explanation techniques and visualisation as well as the code to generate the graphs.

## Code Structure
* The class generate_explanations.py contains the functions to produce the different explantions for a given dataset and machine learning model. 
    * The parameters dataset_names and models could be modified to change the dataset and model uses as classifier.
    * This class calls the function from the class VisualisationExplanation.py that computes the explanations and produces the different visualisation.
* The class generate_results.py is the main class to take the results from the user study and return .csv files with the different metrics and demographic information. Once the different csv files are produced, you can run:
    * analyse_data.r to train a linear model and generate the different ANOVA tables for Table 4, 5, and 6.
    * generate_final_graph.py to generate the different graphs from Figure 4 to 8.

## Data processing
You may find in the class prepare_dataset.py the different functions we used to process the datasets.
For Compas:
* In order to generate explanations meaningful to the users, we removed some features and keep this subset of features {Gender, Age, Race, Juvenile felony count, Juvenile misdemeanor count, Priors count, Charge degree, Charge description}. We also removed 508 individuals having a charge description that occurred less than 5 times in the whole dataset.

## Questionnaires Used
In our survey, we ask the online users to complete three different questionnaires, each one evaluating a given criteria. We present in this section the question and where each questionnaire comes from. 

### Satisfaction Scale
We now present the questions to evaluate the users' perceived satisfaction with the system from Hoffman et al. This questionnaire is composed of 8 questions:
1. From the explanation, I understand how the tool works.
1. This explanation of how the tool works is satisfying.
1. This explanation of how the tool work has sufficient detail.
1. This explanation of how the tool work seems complete.
1. This explanation of how the tool work tells me how to use it.
1. This explanation of how the tool work is useful to my goals.
1. This explanation of the tool shows me how accurate the tool is.
1. This explanation lets me judge when I should trust and not trust the tool.

For each of these questions, Hoffman et al. recommend these 5 Likert scales:
* I disagree strongly & I disagree somewhat & I'm neutral about it & I agree somewhat &  I agree strongly

### Trust Scale
We first use the Cahour-Forzy scale composed of these 4 items with a 7 Likert scale as follows:
* What is your confidence in the tool? Do you have a feeling of trust in it?
    * **I do not trust it at all.** & 2 & 3 & 4 & 5 & 6 & **I trust it completely.**
* Are the actions of the tool predictable?
    * **It is not at all predictable.** & 2 & 3 & 4 & 5 & 6 & **It is completely predictable.**
* Is the tool reliable?
    * **It is not at all reliable.** & 2 & 3 & 4 & 5 & 6 & **It is completely reliable.**
* Is the tool efficient at what it does?
    * **It is not at all efficient.** & 2 & 3 & 4 & 5 & 6 & **It is completely efficient.**

### Understanding Scale
We now present the questions to evaluate the users' perceived understanding of the system from Madsen and Gregor. This questionnaire is composed of 8 questions:
1. The system uses appropriate methods to reach decisions.
1. The system has sound knowledge about this type of problem built into it.
1. The advice the system produce is as good as that which a highly competent person could produce.
1. The system makes use of all the knowledge and information available to it to produce its solution to the problem.
1. I know what will happen the next time I use the system because I understand how it behaves.
1. I understand how the system will assist me with decisions I have to make.
1. Although I may not know exactly how the system works, I know how to use it to make decisions about the problem.
1. It is easy to follow what the system does.

For each of these questions, Madsen and Gregor recommend these 5 Likert scales:
* I disagree strongly & I disagree somewhat & I'm neutral about it & I agree somewhat &  I agree strongly 

## Introduction of the domains and comprehension questions:
* We first introduce the recidivism domain through this paragraph that comes with a comprehension question for which the right answers is "Help the judge decide whether to release a prisoner."
![image](compas_verification1.png)
* Similarly, we present the obesity risk background and ask this question to verify the users comprehension:
![image](obesity_verification1.png)
* Finally, we measure the users comprehension of how an AI predicts based on these two paragraph and question for respectively the COMPAS and Obesity datasets: 
![image](compas_verification2.png)
![image](obesity_verification2.png)

## Online Resources
* The CSV file for the Obesity dataset can be found at this link on [Kaggle](https://www.kaggle.com/code/mpwolke/obesity-levels-life-style) and has been introduced in this [paper](https://doi.org/10.1016/j.dib.2019.104344).
* For COMPAS, the dataset can be found [here](https://www.kaggle.com/datasets/danofer/compass).
