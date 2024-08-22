import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def create_boxplot(data: pd.DataFrame, x: str, y: str, hue: str, xticks: list, 
                   x_labels: list, x_label: str, y_label: str, 
                   filename: str, legend: str) -> None:
    """
    Generate and save a boxplot with exterior information.

    Args:
        data (pd.DataFrame): The data to use for the boxplot.
        x (str): The column in the data to use for the x-axis.
        y (str): The column in the data to use for the y-axis.
        hue (str): The column in the data to use for the hue.
        xticks (list): The x-axis tick locations.
        x_labels (list): The x-axis tick labels.
        x_label (str): The x-axis label.
        y_label (str): The y-axis label.
        filename (str): The name of the file to save the image to.
        legend (str): The title of the legend.
    """
    fig, ax = plt.subplots()
    plt.figure(figsize=(18,8))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    sns.boxplot(data=data, x=x, y=y, hue=hue, palette="Set1", width=0.6)
    if legend != None:
        plt.legend(fontsize=25, loc=(1.04, 0), title=legend, title_fontsize=28)
    plt.xlabel(x_label, fontsize=25)
    plt.ylabel(y_label, fontsize=25)
    plt.xticks(xticks, labels=x_labels, fontsize=20)
    plt.yticks(fontsize=20)
    fig.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show(block=False)
    plt.pause(0.1)
    plt.close('all')

        
metrics = [('b_recall_understanding', 'Score'), 
        ('b_precision_understanding', 'Score'),
        ('perceived_understanding_in_xai', 'Immediate Understanding'),
        ('b_trust', 'Follow Prediction'), 
        ('increased_trust_after_xai', 'Delta Confidence')]

obesity_df = pd.read_csv("./user_study/results_Obesity.csv")
compas_df = pd.read_csv("./user_study/results_Compas.csv")
result_df = pd.concat([obesity_df, compas_df])
sns.set(style="white")

result_df = result_df.reset_index()

for metric in metrics:
        print(metric)
        result_df_temp = result_df
        filename = "./user_study/boxplot/" + metric[0].replace(' ', '_')

        if "understanding" in metric[0]:
            result_df_temp['local_surrogate'].replace({0:"Feature Attribution", 1:"Rules", 2: "Counterfactual", 3: "No Explanation"}, inplace = True)
            create_boxplot(result_df_temp, "domain", metric[0], "local_surrogate", 
                        [0, 1], ["Obesity", "Recidivism"], 
                        "Domain", metric[1], filename + "_surrogate_domain.png", "Explanation Technique")
            
        elif metric[0] == 'b_trust':
            # We remove non zero and null value that are present when the participants
            # did not have to change their response
            nan_idx = pd.isnull(result_df_temp['b_trust']).to_numpy().nonzero()[0]
            result_df_temp = result_df_temp.drop(index = nan_idx)
            result_df_temp_obesity = result_df_temp.loc[result_df_temp['domain'] == "Obesity"]
            result_df_temp_obesity.drop(result_df_temp_obesity[result_df_temp_obesity.representation == 2].index, inplace=True)
            result_df_temp_obesity['representation'].replace({0:"Graphical", 1:"Textual"}, inplace = True)
            
            create_boxplot(result_df_temp_obesity, "local_surrogate", metric[0], "representation", 
                        [0, 1, 2], ["Feature Attribution", "Rules", "Counterfactual"], 
                        "Explanation Technique", metric[1], filename + "_mix.png", "Representation")
            
        else:
            result_df_temp['representation'].replace({0:"Graphical", 1:"Textual", 2: "No Explanation"}, inplace = True)
            create_boxplot(result_df_temp, "domain", metric[0], "representation", 
                        [0, 1], ["Obesity", "Recidivism"], 
                        "Domain", metric[1], filename + "_representation_domain.png", "Representation")