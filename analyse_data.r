data <- read.csv(file = "./user_study/results_Obesity.csv") # conduct the ANOVA analysis on the Obesity dataset only
data <- read.csv(file = "./user_study/results_Compas.csv") # conduct the ANOVA analysis on the Compas dataset only
data <- read.csv(file = "./user_study/results_Obesity_Compas.csv") # conduct the ANOVA analysis on Obesity and Compas datasets
head(data)
# Convert the data to their format
data$Age <- as.numeric(data$Age)
data$representation <- as.factor(data$representation)
data$local_surrogate <- as.factor(data$local_surrogate)
data$Highest_education_level_completed <- as.factor(data$Highest_education_level_completed)
data$sex <- as.factor(data$Sex)

# The metrics on which we conduct the ANOVA Analysis
# Understanding metrics
metrics <- c("understanding", "b_precision_understanding",
            "b_recall_understanding", "b_top_understanding",
            "perceived_understanding_in_xai")
# Confidence metrics
metrics <- c("trust", "b_trust", "b_ord_trust",
            "increased_trust_after_xai", "perceived_trust_in_xai",
            "inc_tru_same", "inc_tru_diff")
# Other metrics removed from the paper
metrics <- c("satisfaction", "Duration_in_seconds", "time")


data <- data[sample(nrow(data)), ] # We randomise the dataset
for (metric in metrics) {
    lm1 <- lm(paste0(metric, "~", "Age + local_surrogate +
            representation + Sex + 
            Highest_education_level_completed +
            local_surrogate:representation"), 
            data = data)
    my_lm_anova <- anova(lm1)
    print(my_lm_anova)
}
