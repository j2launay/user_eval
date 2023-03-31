import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def add_exterior_information(data, x, y, hue, xticks, x_labels, x_label, y_label, filename, legend):
        fig, ax = plt.subplots()
        plt.figure(figsize=(12,6))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        sns.boxplot(data=data, x=x, y=y, hue=hue)
        plt.xticks(xticks, labels=x_labels, fontsize=18)
        plt.yticks(fontsize=18)
        plt.xlabel(x_label, fontsize=25)#"Surrogates", fontsize=25)
        plt.ylabel(y_label, fontsize=25)#metric[1], fontsize=25)
        plt.legend(fontsize=18, title=legend, title_fontsize=20)#, loc="upper left")
        plt.subplots_adjust(bottom=0.15, top=0.95)
        fig.tight_layout()
        plt.savefig(filename)
        plt.show()
        
        
test = "age"
if "understand" in test:
        metrics = [('b_recall_understanding', 'Recall'), ('b_precision_understanding', 'Precision'),  
                #'b_top_understanding', 
                ('perceived_understanding_in_xai', 'Self-repored Understanding'), ('understanding', 'Post Und.')]
elif "trust" in test: 
        metrics = [('b_trust', 'Behavioural Trust'), ('increased_trust_after_xai', 'Self-reported Trust'), #"b_ord_trust", 
                ('trust', 'S.R.Tru.')]#, 'perceived_trust_in_xai']
else:
        metrics = [('time', 'Task Time'), ('satisfaction', 'Post Sat.')]#, 'Duration_in_seconds']

obesity_df = pd.read_csv("./user_study/results_Obesity.csv")
compas_df = pd.read_csv("./user_study/results_Compas.csv")
result_df = pd.concat([obesity_df, compas_df])
#result_df = obesity_df
sns.set(style="darkgrid") 

#print(result_df)
result_df = result_df.reset_index()
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

if test == "age":#age":
        result_df.drop(result_df[result_df.Age == 'CONSENT_REVOKED'].index, inplace=True)
        target_column = result_df['Age'].replace('CONSENT_REVOKED', 2)
        target_column = target_column.astype('int')
        target_column = pd.cut(target_column,
                bins=[5, 20, 30, 40, 60], 
                labels=["< 20", "20 < 30", "30 < 40", "40 >"])
        result_df['Age_bin'] = target_column
for metric in metrics:
        print(metric)
        result_df_temp = result_df
        #sns.swarmplot([representation_picture[metric].tolist(), representation_text[metric].tolist(), 
        #                   representation_control[metric].tolist()], color= "white", hue="domain")
        #sns.violinplot([representation_picture[metric], representation_text[metric], 
        #                    representation_control[metric]], showmedians=True, hue="domain")#, inner="points")
        
        if metric[0] == 'b_trust':
                nan_idx = pd.isnull(result_df_temp['b_trust']).to_numpy().nonzero()[0]
                result_df_temp = result_df_temp.drop(index = nan_idx)
                
        """sns.swarmplot(data=result_df_temp, x="representation", y=metric, hue="domain", color="white")
        sns.violinplot(data=result_df_temp, x="representation", y=metric, hue="domain", showmedians=True)
        plt.xticks([0, 1, 2], labels=["image", "text", "control"])
        plt.xlabel("Groups")
        plt.ylabel("measurements")
        plt.title(metric)
        plt.tight_layout
        plt.savefig("./figures/" + metric.replace(' ', '_') + "_representation.png")
        plt.show()
        #sns.swarmplot([linear_result[metric].tolist(), rule_result[metric].tolist(), 
        #                   counterfactual_result[metric].tolist(), control_group[metric].tolist()], color= "white")
        #sns.violinplot([linear_result[metric].tolist(), rule_result[metric].tolist(), 
        #                    counterfactual_result[metric].tolist(), control_group[metric].tolist()], showmedians=True)#, inner="points")
        
        sns.swarmplot(data=result_df_temp, x="local_surrogate", y=metric, hue="domain", color="white")
        sns.violinplot(data=result_df_temp, x="local_surrogate", y=metric, hue="domain", showmedians=True)
        plt.xticks([0, 1, 2, 3], labels=["Linear", "Rules", "Counterfactual", "control"])
        plt.xlabel("Groups")
        plt.ylabel("measurements")
        plt.title(metric)
        plt.tight_layout
        plt.savefig("./figures/" + metric.replace(' ', '_') + "_surrogate.png")
        plt.show()"""
        filename = "./user_study/boxplot/" + metric[0].replace(' ', '_')
        if "understand" not in test and "trust" not in test:
                add_exterior_information(result_df_temp, "Age_bin", metric[0], "domain",
                                         [0, 1, 2, 3], 
                                         ["< 20", "20 < 30", "30 < 40", "40 >"],
                                         "Age", metric[1], filename + "_age.png", "dataset")
        else:
                result_df_temp_obesity = result_df_temp.loc[result_df_temp['domain'] == "Obesity"]
                #result_df_temp_obesity.drop(result_df_temp_obesity[result_df_temp_obesity.representation == 2].index, inplace=True)
                #result_df_temp_obesity['representation'].replace({0:"Image", 1:"Text"}, inplace = True)
                result_df_temp_obesity['representation'].replace({0:"Image", 1:"Text", 2: "Control"}, inplace = True)
                print(result_df_temp_obesity['representation'])
                add_exterior_information(result_df_temp_obesity, "local_surrogate", metric[0], "representation", 
                                [0, 1, 2, 3], ["Linear", "Rules", "Counterfactual", "Control"], 
                                #[0, 1, 2], ["Linear", "Rules", "Counterfactual"], 
                                "Surrogate", metric[1], filename + "_mix.png", "representation")
                """add_exterior_information(result_df_temp, "Age_bin", metric[0], "domain",
                                         [0, 1, 2, 3], 
                                         ["< 20", "20 < 30", "30 < 40", "40 >"],
                                         "Age", metric[1], filename + "_age.png", "dataset")"""
                
                add_exterior_information(result_df_temp, "local_surrogate", metric[0], "domain", 
                                [0, 1, 2, 3], ["Linear", "Rules", "Counterfactual", "Control"], 
                                "Surrogate", metric[1], filename + "_surrogate.png", "dataset")

                add_exterior_information(result_df_temp, "representation", metric[0], "domain", 
                                [0, 1, 2], ["Image", "Text", "Control"], 
                                "Representation", metric[1], filename + "_representation.png", "dataset")
