library(dplyr)     
library(MASS)        
library(ggplot2)    
library(knitr)       
library(brant)   
library(ggeffects)   
library(nnet)        
library(stargazer) 
library(ordinal)    
library(lmtest)    
library(clubSandwich)


#Data Loading and Preparation

file_path <- 'Long_Format_Data.csv'
long_data <- read.csv(file_path, sep = ',')


prepared_data <- long_data %>%
  # Remove rows where a choice wasn't made (if any)
  filter(!is.na(Model_Choice)) %>%
  mutate(
    #Convert Task_Complexity to a factor with descriptive labels
    Task_Complexity = factor(Task_Complexity,
                             levels = c(0, 1),
                             labels = c("Simple", "Complex")),
    
    #Create the ordered factor for the Dependent Variable
    AI_Model_Choice_Ordinal = factor(Model_Choice,
                                     levels = c(1, 2, 3),
                                     labels = c("Ultra", "Core", "Nano"),
                                     ordered = TRUE),
    
    # Convert CO2_Disclosure to a factor with descriptive labels
    CO2_Disclosure = factor(CO2_Disclosure,
                            levels = c(0, 1),
                            labels = c("No Label", "CO2 Label")),
    

    Env_Concern = as.numeric(Environmental_importance_score),
    
    # Convert character columns to factors for the control model
    AGE_Group = as.factor(AGE_Group),
    Gender = as.factor(Gender)
  )

print("--- Prepared data structure ---")
str(prepared_data)


# Baseline Behaviour Analysis (Control Group) for H1 & H2

print("--- Baseline Analysis (H1 & H2) ---")

# Filter for the control group (no CO2 disclosure)
control_group_data <- prepared_data %>% filter(CO2_Disclosure == "No Label")

# H1: Choice distribution for simple tasks in the control group
simple_table <- table(control_group_data$AI_Model_Choice_Ordinal[control_group_data$Task_Complexity == "Simple"])
print("Choice Distribution for Simple Tasks (Control Group):")
print(simple_table)
simple_chi_test <- chisq.test(simple_table)
print("Chi-Square Test for H1:")
print(simple_chi_test)

# H2: Choice distribution for complex tasks in the control group
complex_table <- table(control_group_data$AI_Model_Choice_Ordinal[control_group_data$Task_Complexity == "Complex"])
print("Choice Distribution for Complex Tasks (Control Group):")
print(complex_table)
complex_chi_test <- chisq.test(complex_table)
print("Chi-Square Test for H2:")
print(complex_chi_test)


# Ordered Logit Regression Modeling

print("--- Ordered Logit Regression (H3, H4, H5) ---")

#Helper function to get p-values from the model summary for polr models
pr <- function(model) {
  ctable <- coef(summary(model))
  p <- pnorm(abs(ctable[, "t value"]), lower.tail = FALSE) * 2
  ctable <- cbind(ctable, "p value" = p)
  return(ctable)
}

# Model 1: Main Effects (H3)
model1 <- polr(AI_Model_Choice_Ordinal ~ CO2_Disclosure + Task_Complexity, data = prepared_data, Hess = TRUE)
print("--- Model 1: Main Effects (H3) ---")
print(kable(pr(model1), digits=4))

# Model 2: Disclosure x Task Complexity Interaction (H4)
model2 <- polr(AI_Model_Choice_Ordinal ~ CO2_Disclosure * Task_Complexity, data = prepared_data, Hess = TRUE)
print("--- Model 2: Disclosure x Task Complexity Interaction (H4) ---")
print(kable(pr(model2), digits=4))

# Model 3: Disclosure x Environmental Concern Interaction (H5)
model3 <- polr(AI_Model_Choice_Ordinal ~ CO2_Disclosure * Env_Concern + Task_Complexity, data = prepared_data, Hess = TRUE)
print("--- Model 3: Disclosure x Environmental Concern Interaction (H5) ---")
print(kable(pr(model3), digits=4))


#Proportional Odds Assumption Check

print("--- Brant Test for Proportional Odds Assumption ---")

brant_test_result <- try(brant(model2), silent = TRUE)
if(inherits(brant_test_result, "try-error")) {
  print("Brant test could not be performed. This can happen with complex models or perfect separation.")
} else {
  print(brant_test_result)
}

print("--- Plotting Interaction Effects ---")

h4_effects <- ggpredict(model2, terms = c("Task_Complexity", "CO2_Disclosure"))
plot_h4 <- plot(h4_effects) +
  labs(
    title = "Interaction: Disclosure and Task Complexity (H4)",
    subtitle = "Predicted Probability of AI Model Choice",
    x = "Task Complexity",
    y = "Predicted Probability",
    colour = "Disclosure",
    fill = "Disclosure"
  ) +
  theme_minimal()
print(plot_h4)

h5_effects <- ggpredict(model3, terms = c("Env_Concern [all]", "CO2_Disclosure"))
plot_h5 <- plot(h5_effects) +
  labs(
    title = "Interaction: Disclosure and Environmental Concern (H5)",
    subtitle = "Predicted Probability of AI Model Choice",
    x = "Environmental Concern Score",
    y = "Predicted Probability",
    colour = "Disclosure",
    fill = "Disclosure"
  ) +
  theme_minimal()
print(plot_h5)


#Robustness Checks

#Robustness Check 1: Multinomial Logit Model
print("--- Robustness Check 1: Multinomial Logit Model ---")


multinom_model <- multinom(AI_Model_Choice_Ordinal ~ CO2_Disclosure * Task_Complexity, data = prepared_data)

multinom_summary <- summary(multinom_model)
coeffs <- multinom_summary$coefficients
std_errs <- multinom_summary$standard.errors
z_scores <- coeffs / std_errs
p_values <- (1 - pnorm(abs(z_scores), 0, 1)) * 2

print("--- Multinomial Model Coefficients ---")
print(kable(coeffs, digits=4))
print("--- Multinomial Model p-values ---")
print(kable(p_values, digits=4))


#Robustness Check 2: Ordered Logit with Control Variables
print("--- Robustness Check 2: Ordered Logit with Controls ---")

#Add demographic and attitudinal variables to see if the key interaction remains significant.
model_with_controls <- polr(AI_Model_Choice_Ordinal ~ CO2_Disclosure * Task_Complexity +
                              AI_Familiarity_score + AI_Usage_Freq + Trust_overall +
                              Env_Concern + AGE_Group + Gender,
                            data = prepared_data, Hess = TRUE)

print(kable(pr(model_with_controls), digits=4))


#Robustness Check 3: Mixed-Effects Ordered Logit Model (using Wald tests)

clmm_data <- dplyr::select(
  prepared_data,
  AI_Model_Choice_Ordinal,
  CO2_Disclosure,
  Task_Complexity,
  Env_Concern,
  Response_ID
) %>%
  na.omit()

#Fit the cumulative link mixed model
library(ordinal)
model_clmm <- clmm(
  AI_Model_Choice_Ordinal ~ CO2_Disclosure * Task_Complexity + Env_Concern +
    (1 | Response_ID),         
  data    = clmm_data,
  Hess    = TRUE,            
  nAGQ    = 10,              
  control = clmm.control(optimizer = "Nelder-Mead")
)

#Display the built-in Wald z-tests / p-values
summary(model_clmm)


simple_table_df <- as.data.frame(simple_table)
colnames(simple_table_df) <- c("Model Choice", "Frequency")
complex_table_df <- as.data.frame(complex_table)
colnames(complex_table_df) <- c("Model Choice", "Frequency")

# 7. Generate tables with HTML output for in final doc

stargazer(
  model1,
  type = "html",
  title = "Ordered Logit Results (Model 1): Main Effects of Disclosure and Task Complexity",
  dep.var.labels = "AI Model Choice (Nano > Core > Ultra)",
  covariate.labels = c("CO2 Disclosure (Label)", "Task Complexity (Complex)"),
  omit = "^zeta",
  df = FALSE,
  notes = "Testing Hypothesis H3",
  out = "table_model1_h3.html"
)

stargazer(
  model2,
  type = "html",
  title = "Ordered Logit Results (Model 2): Interaction of Disclosure and Task Complexity",
  dep.var.labels = "AI Model Choice (Nano > Core > Ultra)",
  covariate.labels = c("CO2 Disclosure (Label)", "Task Complexity (Complex)", "Disclosure x Task Complexity"),
  omit = "^zeta",
  df = FALSE,
  notes = "Testing Hypothesis H4",
  out = "table_model2_h4.html"
)


stargazer(
  model3,
  type = "html",
  title = "Ordered Logit Results (Model 3): Interaction of Disclosure and Environmental Concern",
  dep.var.labels = "AI Model Choice (Nano > Core > Ultra)",
  covariate.labels = c("CO2 Disclosure (Label)", "Environmental Concern", "Task Complexity (Complex)", "Disclosure x Environmental Concern"),
  omit = "^zeta",
  df = FALSE,
  notes = "Testing Hypothesis H5",
  out = "table_model3_h5.html"
)


stargazer(
  model_with_controls,
  type = "html",
  title = "Ordered Logit Results: Full Model with Control Variables",
  dep.var.labels = "AI Model Choice (Nano > Core > Ultra)",
  covariate.labels = c(
    "CO2 Disclosure (Label)",
    "Task Complexity (Complex)",
    "AI Familiarity",
    "AI Usage Frequency",
    "Trust Overall",
    "Environmental Concern",
    "Age Group: 25-34",
    "Age Group: 35-49",
    "Age Group: 50-64",
    "Age Group: 65+",
    "Gender: Female",
    "Disclosure x Task Complexity"
  ),
  omit = "^zeta",
  df = FALSE,
  notes = "This model includes all covariates as a robustness check.",
  out = "table_full_model_controls.html"
)

stargazer(
  multinom_model,
  type = "html",
  title = "Multinomial Logit Results (Robustness Check for H4)",
  notes = "Baseline outcome category is 'Ultra'.",
  dep.var.labels.include = FALSE, 
  
  
  covariate.labels = c(
    "CO₂ Disclosure (Label)",
    "Task Complexity (Complex)",
    "Disclosure × Task Complexity"
  ),
  
  out = "table_multinomial_robustness.html"
)

clmm_summary <- summary(model_clmm)
clmm_coeffs <- as.data.frame(clmm_summary$coefficients)

fixed_effects <- clmm_coeffs[c("CO2_DisclosureCO2 Label", "Task_ComplexityComplex", "Env_Concern", "CO2_DisclosureCO2 Label:Task_ComplexityComplex"), ]

rownames(fixed_effects) <- c(
  "CO2 Disclosure (Label)",
  "Task Complexity (Complex)",
  "Environmental Concern",
  "Disclosure x Task Complexity"
)

fixed_effects$'p value' <- fixed_effects$'Pr(>|z|)'
fixed_effects$Estimate_str <- paste0(
  round(fixed_effects$Estimate, 3),
  ifelse(fixed_effects$'p value' < 0.001, "***",
         ifelse(fixed_effects$'p value' < 0.01,  "**",
                ifelse(fixed_effects$'p value' < 0.05,  "*", "")))
)
fixed_effects$'Std. Error_str' <- paste0("(", round(fixed_effects$'Std. Error', 3), ")")


final_clmm_table_df <- data.frame(
  "Predictor" = rownames(fixed_effects),
  "Estimate" = fixed_effects$Estimate_str,
  "Std. Error" = fixed_effects$'Std. Error_str'
)

library(knitr)
clmm_html_output <- kable(final_clmm_table_df, 
                          format = "html", 
                          escape = FALSE, 
                          align = "lrr",
                          col.names = c("", "Estimate", "Std. Error"),
                          caption = "<b>Table 7: Mixed-Effects Ordered Logit Results (CLMM)</b>")


file_name <- "table_clmm_final.html"
writeLines(clmm_html_output, file_name)
