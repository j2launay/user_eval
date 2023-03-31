data <- read.csv(file = "./user_study/all_results_r.csv")
data <- read.csv(file = "./user_study/Obesity_results_r.csv")
data <- read.csv(file = "./user_study/results_Obesity.csv")
data <- read.csv(file = "./user_study/results_Compas.csv")
head(data)
data$Age <- as.numeric(data$Age)
data$representation <- as.factor(data$representation)
data$local_surrogate <- as.factor(data$local_surrogate)
data$Highest_education_level_completed <- as.factor(data$Highest_education_level_completed)
data$sex <- as.factor(data$Sex)

metrics <- c("understanding", "b_precision_understanding",
            "b_recall_understanding", "b_top_understanding",
            "perceived_understanding_in_xai")
metrics <- c("trust", "b_trust", "b_ord_trust",
            "increased_trust_after_xai", "perceived_trust_in_xai")
metrics <- c("satisfaction",
            "Duration_in_seconds", "time")


data <- data[sample(nrow(data)), ]
for (metric in metrics) {
    lm1 <- lm(paste0(metric, "~", "Age +
            local_surrogate +
            representation +
            Sex + Highest_education_level_completed +
            local_surrogate:representation
            "), data = data)
    my_lm_anova <- anova(lm1)
    print(my_lm_anova)
}


data["domain"][data["domain"] == "Compas"] <- 0
data["domain"][data["domain"] == "Obesity"] <- 1
data$domain <- as.factor(data$domain)

for (metric in metrics) {
    lm1 <- lm(paste0(metric, "~", "representation_0 +
            representation_1 + representation_2 +
            local_surrogate_0 + local_surrogate_1 +
            local_surrogate_2 + local_surrogate_3 +
        Highest_education_Undergraduate.degree..BA.BSc.other. +
        Highest_education_Technical.community.college +
        Highest_education_High.school.diploma.A.levels +
        Highest_education_Graduate.degree..MA.MSc.MPhil.other. +
        Highest_education_Doctorate.degree..PhD.other. +
        Highest_education_CONSENT_REVOKED +
            domain + Age + Sex +
            representation_0:local_surrogate_0 +
            representation_0:local_surrogate_1 +
            representation_0:local_surrogate_2 +
            representation_0:local_surrogate_3 + 
            representation_1:local_surrogate_0 +
            representation_1:local_surrogate_1 +
            representation_1:local_surrogate_2 +
            representation_1:local_surrogate_3 + 
            representation_2:local_surrogate_0 +
            representation_2:local_surrogate_1 +
            representation_2:local_surrogate_2 +
            representation_2:local_surrogate_3
            "), data = data)
    my_lm_anova <- anova(lm1)
    #ggplot(data, aes(x = "representation", y = metric)) + geom_boxplot()
    #ggsave(paste("./user_study/boxplot/", metric, ".png", sep = ""),
    #            width = 4, height = 3, dpi = 1000)
    print(my_lm_anova)
}

for (metric in metrics) {
    lm1 <- lm(paste0(metric, "~", "representation_0 +
            local_surrogate_0 + local_surrogate_1 +
            local_surrogate_2 + local_surrogate_3 + Age + Sex 
            "), data = data)
    my_lm_anova <- anova(lm1)
    #ggplot(data, aes(x = "representation", y = metric)) + geom_boxplot()
    #ggsave(paste("./user_study/boxplot/", metric, ".png", sep = ""),
    #            width = 4, height = 3, dpi = 1000)
    print(my_lm_anova)
}


for (metric in metrics) {
    lm1 <- lm(paste0(metric, "~", "representation_0 +
            representation_1 + representation_2 +
            local_surrogate_0 + local_surrogate_1 +
            local_surrogate_2 + local_surrogate_3 +
        Highest_education_Undergraduate.degree..BA.BSc.other. +
        Highest_education_Technical.community.college +
        Highest_education_High.school.diploma.A.levels +
        Highest_education_Graduate.degree..MA.MSc.MPhil.other. +
        Highest_education_Doctorate.degree..PhD.other. +
        Highest_education_CONSENT_REVOKED +
            domain + Age + Sex +
            representation_0:local_surrogate_0 +
            representation_0:local_surrogate_1 +
            representation_0:local_surrogate_2 +
            representation_0:local_surrogate_3 + 
            representation_1:local_surrogate_0 +
            representation_1:local_surrogate_1 +
            representation_1:local_surrogate_2 +
            representation_1:local_surrogate_3 + 
            representation_2:local_surrogate_0 +
            representation_2:local_surrogate_1 +
            representation_2:local_surrogate_2 +
            representation_2:local_surrogate_3
            "), data = data)
    my_lm_anova <- anova(lm1)
    #ggplot(data, aes(x = "representation", y = metric)) + geom_boxplot()
    #ggsave(paste("./user_study/boxplot/", metric, ".png", sep = ""),
    #            width = 4, height = 3, dpi = 1000)
    print(my_lm_anova)
}
