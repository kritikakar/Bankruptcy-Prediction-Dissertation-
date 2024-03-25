library("farff")

setwd("C:/Users/kriti/OneDrive/Documents/Maths Diss/Polish Bankruptcy")

# importing the dataset

data <- readARFF("3year.arff")

# observing the correlation between independent variables

library(caret)

correlation_matrix <- cor(data[, -ncol(data)], use = "complete.obs", method = "pearson")

print(correlation_matrix)

high_correlation_pairs <- findCorrelation(correlation_matrix, cutoff = 0.75, names = FALSE)

# removing highly correlated columns

data1 <- data[, -high_correlation_pairs]

# observing number of missing data points by column

na_col <- colSums(is.na(data1))

print(na_col)

# removing attribute 37 due to a large proportion of missing values

data1 <- data1[, !colnames(data1) %in% "Attr37"]

# removing attribute 27 due to noise/not theoretically relevant/missing values

data1 <- data1[, !colnames(data1) %in% "Attr27"]

# removing rows with 5 or more missing values

data2 <- data1[rowSums(is.na(data1)) < 5, ]

na_col2 <- colSums(is.na(data2))

print(na_col2)

# Checking for outliers before attempting imputation  

summary(data2)

# removing a duplicate row

data2 <- data2[-1051, ]

library(ggplot2)

for (col in names(data2)) {
  if (col != "class") { 
    plot_data <- data2[, c(col, "class")] 
    plot <- ggplot(plot_data, aes(x = class, y = .data[[col]])) + geom_boxplot() + labs(title = paste("Box plot of", col, "by class"))
    print(plot)
  }
}

summary(data2)

# removing outliers

# most outliers exist for class 0, so removing does not affect analysis

data2 <- subset(data2, data2[, "Attr5"] != -11903000.0)

data2 <- subset(data2, data2[, "Attr6"] != -508.1200)

data2 <- subset(data2, data2[, "Attr6"] != -486.7200)

data2 <- subset(data2, data2[, "Attr13"] != 4972.00000)

data2 <- subset(data2, data2[, "Attr15"] != 10236000)

data2 <- subset(data2, data2[, "Attr17"] != 29642.000)

data2 <- subset(data2, data2[, "Attr17"] != 26382.000)

data2 <- subset(data2, data2[, "Attr17"] != 11491.000)

data2 <- subset(data2, data2[, "Attr21"] != 29907.0000)

data2 <- subset(data2, data2[, "Attr25"] != -500.75000)

data2 <- subset(data2, data2[, "Attr25"] != -4.6629e+02)

data2 <- subset(data2, data2[, "Attr32"] != 6674200.00)

data2 <- subset(data2, data2[, "Attr32"] != 6674200.00)

data2 <- subset(data2, data2[, "Attr34"] != 1960.300)

data2 <- subset(data2, data2[, "Attr38"] != -479.73000)

data2 <- subset(data2, data2[, "Attr38"] != -4.4586e+02)

data2 <- subset(data2, data2[, "Attr39"] != -551.11000)

data2 <- subset(data2, data2[, "Attr39"] != -527.95000)

data2 <- subset(data2, data2[, "Attr39"] != 293.15000)

data2 <- subset(data2, data2[, "Attr41"] != 2.8877e+05)

data2 <- subset(data2, data2[, "Attr45"] != -28810.0000)

data2 <- subset(data2, data2[, "Attr45"] != -74385.0000)

data2 <- subset(data2, data2[, "Attr47"] != 2591100.00)

data2 <- subset(data2, data2[, "Attr47"] != 1848800.00)

data2 <- subset(data2, data2[, "Attr59"] != 7617.3000)

data3 <- data2

# attempting imputation

na_col3 <- colSums(is.na(data3))

print(na_col3)

sum(is.na(data3))

# imputing values using the Random Forest Approach

library(missForest)
library(doParallel)

registerDoParallel(cores = 5)

# setting seed for reproducibility

set.seed(416878)

imputed_data <- missForest(data3, verbose = TRUE, parallelize = "variables")

data4 <- imputed_data$ximp

# most missing values are in Attr24, Attr28, Attr53

summary(data4$Attr24)

summary(data3$Attr24)

summary(data4$Attr28)

summary(data3$Attr28)

summary(data4$Attr53)

summary(data3$Attr53)

# Comparing the summary statistics for these columns, look very similar

imputed_data$OOBerror

# error is quite low

final_data <- data4


#############################

for (i in 1:31) {
  print(colnames(final_data)[i])
  print(cor.test(final_data[,i], as.numeric(final_data[,32])))
  
}

# higher correlation for Attr3, Attr6, Attr15, Attr17, Attr24, Attr25, Attr26, Attr33, Attr34, Attr35, Attr38, Attr39, Attr48


#############################


# logistic regression model 1: manual variable selection


library(tidyverse)

library(caret)

library(caTools)

# creating a training and testing dataset with the same ratio as original dataset

set.seed(416878)

train <- createDataPartition(final_data$class, p = 0.8, list = FALSE)
training_data <- final_data[train, ]
testing_data <- final_data[-train, ]

# fitting a logistic regression with all variables

logistic_31 <- glm(training_data$class ~ ., data = training_data, family = binomial(link = "logit"))

# probably have too many variables

# performing manual backward selection by removing the covariates which do not explain the dependent variable

summary(logistic_31)

logistic_23 <- glm(class ~ Attr3 + Attr9 + Attr13 + Attr15 + Attr17 + Attr23 + Attr24 + Attr26 + Attr28 + Attr32 + Attr33 + Attr34 + Attr35 + Attr36 + Attr39 + Attr40 + Attr41 + Attr45 + Attr45 + Attr47 + Attr48 + Attr58 + Attr59 + Attr61, data = training_data, family =  binomial(link = "logit"))

summary(logistic_23)

logistic_17 <- glm(class ~ Attr3 + Attr9 + Attr13 + Attr15 + Attr17 + Attr23 + Attr24 + Attr26 + Attr32 + Attr33 + Attr34 + Attr35 + Attr36 + Attr39 + Attr41 + Attr48 + Attr58, data =  training_data, family = binomial(link = "logit"))

summary(logistic_17)

logistic_13 <- glm(class ~ Attr3 + Attr9 + Attr13 + Attr15 + Attr23 + Attr26 + Attr32 + Attr33 + Attr34 + Attr35 + Attr36 + Attr48 + Attr58, data =  training_data, family = binomial(link = "logit"))

summary(logistic_13)

# AIC is worse now

logistic_15 <- glm(class ~ Attr3 + Attr9 + Attr13 + Attr15 + Attr23 + Attr24 + Attr26 + Attr32 + Attr33 + Attr34 + Attr35 + Attr36 + Attr41 + Attr48 + Attr58, data =  training_data, family = binomial(link = "logit"))

summary(logistic_15)

# AIC is the same

# testing model accuracy on testing dataset

evaluation <- function(model, testingdata) {
  predictions <- predict(model, testingdata, type = "response")
  predictions <- ifelse(predictions >= 0.5, 1, 0)
  confusion_matrix <- confusionMatrix(table(predictions, testingdata[, ncol(testingdata)]))
  print(confusion_matrix)
}

evaluation(logistic_13, testing_data)

# does not predict 1s at all

evaluation(logistic_15, testing_data)

# terrible model (accuracy for 0s is fine but does not predict bankruptcy)

evaluation(logistic_17, testing_data)

# again does not predict bankruptcy


#######################################



# trying probit regression with manual variable selection

probit_23 <- glm(class ~ Attr3 + Attr9 + Attr13 + Attr15 + Attr17 + Attr23 + Attr24 + Attr26 + Attr28 + Attr32 + Attr33 + Attr34 + Attr35 + Attr36 + Attr39 + Attr40 + Attr41 + Attr45 + Attr45 + Attr47 + Attr48 + Attr58 + Attr59 + Attr61, data = training_data, family =  binomial(link = "probit"))

summary(probit_23)

probit_16 <- glm(class ~ Attr3 + Attr9 + Attr13 + Attr15 + Attr23 + Attr24 + Attr26 + Attr32 + Attr33 + Attr34 + Attr35 + Attr36 + Attr39 + Attr41 + Attr48 + Attr58, data =  training_data, family = binomial(link = "probit"))

summary(probit_16)

probit_12 <- glm(class ~ Attr9 + Attr13 + Attr23 + Attr24 + Attr26 + Attr32 + Attr33 + Attr34 + Attr35 + Attr36 + Attr48 + Attr58, data =  training_data, family = binomial(link = "probit"))

summary(probit_12)

evaluation(probit_12, testing_data)

# still not predicting 1s



######################################



# trying step() and backward selection to select variables

# starting with logistic_31 as the initial model

bwd_sel_logit <- step(logistic_31, direction = "backward")

evaluation(bwd_sel_logit, testing_data)

# not predicting bankruptcy still

probit_31 <- glm(training_data$class ~ ., data = training_data, family = binomial(link = "probit"))

bwd_sel_probit <- step(probit_31, direction = "backward")

evaluation(bwd_sel_probit, testing_data)

# not predicting bankruptcy

# performs better at prediction than manual - so will continue with step()



#######################################



# trying interactions in logit model with 14 variables

logistic_14 <- glm(training_data$class ~ Attr3 + Attr9 + Attr13 + Attr15 + Attr17 + Attr23 + Attr26 + Attr32 + Attr33 + Attr34 + Attr35 + Attr36 + Attr48 + Attr58, data = training_data, family = binomial(link = "logit"))

logistic_14_2 <- glm(training_data$class ~ (Attr3 + Attr9 + Attr13 + Attr15 + Attr17 + Attr23 + Attr26 + Attr32 + Attr33 + Attr34 + Attr35 + Attr36 + Attr48 + Attr58)^2, data = training_data, family = binomial(link = "logit"))

logit_int_1 <- step(logistic_14, direction = "both", scope = formula(logistic_14_2))

summary(logit_int_1)

evaluation(logistic_14, testing_data)

evaluation(logit_int_1, testing_data)

# better model than before but quite complicated and not good with predicting. probably overfitted



#############################################



# using elastic net regularization on 31 variables

library(glmnet)

# creating a function to find the best alpha and lambda inputs for elastic net

opt_inputs <- function(x.train, y.train) {
  alpha_values <- seq(0, 1, by = 0.1)
  all_combinations <- expand.grid(alpha = alpha_values, lambda = numeric(0))
  cv_results <- list()
  
  for (alpha_val in alpha_values) {
    cv_elnet <- cv.glmnet(x.train, y.train, family = "binomial", alpha = alpha_val)
    best_lambda <- cv_elnet$lambda.min
    all_combinations <- rbind(all_combinations, data.frame(alpha = alpha_val, lambda = best_lambda))
    cv_results[[as.character(alpha_val)]] <- cv_elnet
  }
  
  best_combination <- all_combinations[which.min(sapply(cv_results, function(cv) min(cv$cvm))), ]
  
  return(best_combination)
}

# creating new testing and training sets as per the model's syntax

x.train <- as.matrix(training_data[,-ncol(training_data)])
y.train <- as.vector(training_data[,ncol(training_data)])
x.test <- as.matrix(testing_data[,-ncol(testing_data)])
y.test <- as.vector(testing_data[,ncol(testing_data)])

# running the elastic net

elnet_31 <- glmnet(x.train, y.train, family = "binomial", alpha = opt_inputs(x.train, y.train)$alpha, lambda = opt_inputs(x.train, y.train)$lambda)

# evaluating the function by editing the function as per syntax 

evaluation2 <- function(model, x.test, y.test) {
  predictions <- predict(model, newx = x.test, type = "response")
  predictions <- ifelse(predictions >= 0.5, 1, 0)
  confusion_matrix <- confusionMatrix(table(predictions, y.test))
  print(confusion_matrix)
}

evaluation2(elnet_31, x.test, y.test)

coef(elnet_31)

# choosing the best variables and running the elastic net model again

selected_vars <- c("Attr9", "Attr15", "Attr17", "Attr23", "Attr24", "Attr26", "Attr33", "Attr35", "Attr36", "Attr38", "Attr41", "Attr48", "Attr59")
selected_indices <- match(selected_vars, colnames(x.train))
x.train.subset <- x.train[, selected_indices]
x.test.subset <- x.test[, selected_indices]

elnet_13 <- glmnet(x.train.subset, y.train, family = "binomial", alpha = opt_inputs(x.train, y.train)$alpha, lambda = opt_inputs(x.train, y.train)$lambda)

# evaluating the model

evaluation2(elnet_13, x.train.subset, y.train)
evaluation2(elnet_13, x.test.subset, y.test)

# accuracy is similar to logistic with int, not overfitted



##############################################



# random forest 

set.seed(416878)

library(randomForest)

# fitting the random forest model

rf_14 <- randomForest(data = training_data, class ~ Attr3 + Attr9 + Attr13 + Attr15 + Attr17 + Attr23 + Attr26 + Attr32 + Attr33 + Attr34 + Attr35 + Attr36 + Attr48 + Attr58, ntree = 500)

evaluation3 <- function(model, testingdata) {
  predictions <- predict(model, testingdata, type = "response")
  confusion_matrix <- confusionMatrix(table(predictions, testingdata[, ncol(testingdata)]))
  print(confusion_matrix)
}

# evaluating the model and checking for overfitting

evaluation3(rf_14, testing_data)

evaluation3(rf_14, training_data)

# performs very well on training data; might be overfitted

# trying to reduce overfitting

rf_14_2 <- randomForest(data = training_data, class ~ Attr3 + Attr9 + Attr13 + Attr15 + Attr17 + Attr23 + Attr26 + Attr32 + Attr33 + Attr34 + Attr35 + Attr36 + Attr48 + Attr58, ntree = 1000)

evaluation3(rf_14_2, testing_data)

evaluation3(rf_14_2, training_data)

rf_14_3 <- randomForest(data = training_data, class ~ Attr3 + Attr9 + Attr13 + Attr15 + Attr17 + Attr23 + Attr26 + Attr32 + Attr33 + Attr34 + Attr35 + Attr36 + Attr48 + Attr58, ntree = 1500, mtry = 2, max_depth = 5)

evaluation3(rf_14_3, testing_data)

evaluation3(rf_14_3, training_data)

rf_14_4 <- randomForest(data = training_data, class ~ Attr3 + Attr9 + Attr13 + Attr15 + Attr17 + Attr23 + Attr26 + Attr32 + Attr33 + Attr34 + Attr35 + Attr36 + Attr48 + Attr58, ntree = 1500, mtry = 2, max_depth = 5, min_samples_split = 50, min_samples_leaf = 50)

evaluation3(rf_14_4, testing_data)

evaluation3(rf_14_4, training_data)

rf_14_5 <- randomForest(data = training_data, 
                         class ~ Attr3 + Attr9 + Attr13 + Attr15 + Attr17 + Attr23 + Attr26 + Attr32 + Attr33 + Attr34 + Attr35 + Attr36 + Attr48 + Attr58, 
                         ntree = 500,
                         classwt = c(1, 19))

evaluation3(rf_14_5, testing_data)

evaluation3(rf_14_5, training_data)

# no change in overfitting



####################################



# comparing logit, logit with interactions, elastic net, and random forest using ROC

predictions_rf_14 <- predict(rf_14, testing_data, type = "class")

predictions_logit_14 <- predict(logistic_14, testing_data, type = "response")
predictions_logit_14 <- ifelse(predictions_logit_14 > 0.5, 1, 0)

predictions_logit_int_14 <- predict(logit_int_1, testing_data, type = "response")
predictions_logit_int_14 <- ifelse(predictions_logit_int_14 > 0.5, 1, 0)

predictions_elnet_13 <- predict(elnet_13, x.test.subset, type = "response")
predictions_elnet_13 <- ifelse(predictions_elnet_13 > 0.5, 1, 0)


library(pROC)

num.y.test <- as.numeric(y.test)

roc_rf_14 <- roc(num.y.test, as.numeric(predictions_rf_14))
roc_logit_14 <- roc(as.vector(testing_data$class), predictions_logit_14)
roc_logit_int_14 <- roc(as.vector(testing_data$class), predictions_logit_int_14)
roc_elnet_13 <- roc(as.vector(y.test), predictions_elnet_13[, 1])

plot(roc_rf_14, col = "olivedrab3", main = "ROC Curves for Multiple Models")
plot(roc_logit_14, col = "violetred", add = TRUE)
plot(roc_logit_int_14, col = "hotpink4", add = TRUE)
plot(roc_elnet_13, col = "cyan3", add = TRUE)

legend("bottomright", legend = c("Random Forest", "Logistic", "Logistic with Interaction", "Elastic Net"), 
       col = c("olivedrab3", "violetred", "hotpink4","cyan3"), lty = 1)




###################################################




# creating a new training dataset with 50% 1s


ones <- filter(final_data, class == 1)
zeros <- filter(final_data, class == 0)

set.seed(416878)


n_train1_50 <- round(0.8 * nrow(ones))
training1_50 <- sample_n(ones, n_train1_50)


n_train0_50 <- n_train1_50
training0_50 <- sample_n(zeros, n_train0_50)

train50 <- bind_rows(training1_50, training0_50)

test50 <- anti_join(final_data, select(training_data, -class))

# logit model using step()

logit31_50 <- glm(class ~ ., data = train50, family = binomial(link = "logit"))

logitstep_50 <- step(logit31_50, direction = "backward")

logit15_50 <- glm(class ~ Attr29 + Attr61 + Attr53 + Attr28 + Attr13 + Attr23 + Attr48 + Attr3 + Attr25 + Attr6 + Attr34 + Attr39 + Attr36 + Attr40 + Attr26, data = train50, family = binomial(link = "logit"))

# logit with interactions

logit15sq_50 <- glm(class ~ (Attr29 + Attr61 + Attr53 + Attr28 + Attr13 + Attr23 + Attr48 + Attr3 + Attr25 + Attr6 + Attr34 + Attr39 + Attr36 + Attr40 + Attr26)^2, data = train50, family = binomial(link = "logit"))

logit_int_50 <- step(logit15_50, direction = "both", scope = formula(logit15sq_50))

# elastic net

# creating training and testing subsets as per syntax

x_train_50 <- as.matrix(train50[,-ncol(train50)])
y_train_50 <- as.vector(train50[,ncol(train50)])
x_test_50 <- as.matrix(test50[,-ncol(test50)])
y_test_50 <- as.vector(test50[,ncol(test50)])

# fitting the model

elnet31_50 <- glmnet(x_train_50, y_train_50, family = "binomial", alpha = opt_inputs(x_train_50, y_train_50)$alpha, lambda = opt_inputs(x_train_50, y_train_50)$lambda)

# choosing the best coefficients

coef(elnet31_50)

selected_vars_50 <- c("Attr6", "Attr13", "Attr15", "Attr24", "Attr26", "Attr29", "Attr33", "Attr34", "Attr35", "Attr36", "Attr39", "Attr45", "Attr47", "Attr58", "Attr61")

selected_indices_50 <- match(selected_vars_50, colnames(x_train_50))
x_train_50_subset <- x_train_50[, selected_indices_50]
x_test_50_subset <- x_test_50[, selected_indices_50]

# running another model

elnet14_50 <- glmnet(x_train_50_subset, y_train_50, family = "binomial", alpha = opt_inputs(x_train_50, y_train_50)$alpha, lambda = opt_inputs(x_train_50, y_train_50)$lambda)

evaluation2(elnet14_50, as.matrix(x_test_50_subset), as.vector(y_test_50))

# random forest using logistic regression variables

rf15_50 <- randomForest(data = train50, class ~ Attr29 + Attr61 + Attr53 + Attr28 + Attr13 + Attr23 + Attr48 + Attr3 + Attr25 + Attr6 + Attr34 + Attr39 + Attr36 + Attr40 + Attr26, ntree = 500)

# running the random forest again with chosen variables using the variable importance

rf31_50 <- randomForest(data = train50, class ~ ., ntree = 500)

importance(rf31_50)

rf14_50 <- randomForest(data = train50, class ~ Attr6 + Attr13 + Attr15 + Attr24 + Attr26 + Attr29 + Attr33 + Attr34 + Attr35 + Attr36 + Attr39 + Attr45 + Attr47 + Attr58 + Attr61, ntree = 500)

# evaluating all models

evaluation(logitstep_50, test50)

evaluation(logit_int_50, test50)

evaluation3(rf15_50, test50)

evaluation3(rf14_50, test50)

# rf with important variables as per rf31 performs slightly better than with logit variables

# creating a testing subset

ytest_50 <- as.vector(test50[,ncol(test50)])

# using the ROC curve to compare models

library(pROC)

predictions_rf14_50 <- predict(rf14_50, test50, type = "class")
predictions_rf15_50 <- predict(rf15_50, test50, type = "class")

predictions_logit15_50 <- predict(logit15_50, test50, type = "response")
predictions_logit15_50 <- ifelse(predictions_logit15_50 > 0.5, 1, 0)

predictions_logit_int_50 <- predict(logit_int_50, test50, type = "response")
predictions_logit_int_50 <- ifelse(predictions_logit_int_50 > 0.5, 1, 0)

predictions_elnet14_50 <- predict(elnet14_50, x_test_50_subset, type = "response")
predictions_elnet14_50 <- ifelse(predictions_elnet14_50 > 0.5, 1, 0)

num.ytest_50 <- as.numeric(ytest_50)

roc_rf15_50 <- roc(num.ytest_50, as.numeric(predictions_rf15_50))
roc_logit15_50 <- roc(as.vector(test50$class), predictions_logit15_50)
roc_logit_int_50 <- roc(as.vector(test50$class), predictions_logit_int_50)
roc_elnet14_50 <- roc(as.vector(test50$class), predictions_elnet14_50[, 1])

plot(roc_rf15_50, col = "olivedrab3", main = "ROC Curves for Multiple Models")
plot(roc_logit15_50, col = "violetred", add = TRUE)
plot(roc_logit_int_50, col = "hotpink4", add = TRUE)
plot(roc_elnet14_50, col = "cyan3", add = TRUE)

legend("bottomright", legend = c("Random Forest", "Logistic", "Logistic with Interaction", "Elastic Net"), 
       col = c("olivedrab3", "violetred", "hotpink4","cyan3"), lty = 1)



# checking for overfitting

evaluation(logitstep_50, test50)
evaluation(logitstep_50, train50)

evaluation(logit_int_50, test50)
evaluation(logit_int_50, train50)

# not overfitted

evaluation3(rf15_50, test50)
evaluation3(rf15_50, train50)

evaluation3(rf14_50, test50)
evaluation3(rf14_50, train50)

# may be overfitted

importance(rf15_50)

# fitting with top 5 variables

rf5_50 <- randomForest(data = train50, class ~ Attr29 + Attr61 + Attr53 + Attr28 + Attr13, ntree = 500)

evaluation3(rf5_50, test50)

evaluation3(rf5_50, train50)

# not affecting accuracy of training data



###########################################



# using a dataset of 25% ones and 75% zeros

# creating the datasets

set.seed(416878)

n_train1_25 <- round(0.8 * nrow(ones))
training1_25 <- sample_n(ones, n_train1_25)


n_train0_25 <- round((0.75 * n_train1_25) / 0.25)
training0_25 <- sample_n(zeros, n_train0_25)


train25 <- bind_rows(training1_25, training0_25)


test25 <- anti_join(final_data, select(train25, -class))

# logistic regression

logit31_25 <- glm(class ~ ., data = train25, family = binomial(link = "logit"))

logitstep_25 <- step(logit31_25, direction = "backward")

logit15_25 <- glm(class ~ Attr15 + Attr41 + Attr9 + Attr25 + Attr40 + Attr6 + Attr48 + Attr33 + Attr29 + Attr3 + Attr39 + Attr36 + Attr23 + Attr13 + Attr26, data = train25, family = binomial(link = "logit"))

# logistic regression with interactions

logit15sq_25 <- glm(class ~ (Attr15 + Attr41 + Attr9 + Attr25 + Attr40 + Attr6 + Attr48 + Attr33 + Attr29 + Attr3 + Attr39 + Attr36 + Attr23 + Attr13 + Attr26)^2, data = train25, family = binomial(link = "logit"))

logit_int_25 <- step(logit15_25, direction = "both", scope = formula(logit15sq_25))

# elastic net

x_train_25 <- as.matrix(train25[, -ncol(train25)])
y_train_25 <- as.vector(train25[, ncol(train25)])
x_test_25 <- as.matrix(test25[, -ncol(test25)])
y_test_25 <- as.vector(test25[, ncol(test25)])

elnet31_25 <- glmnet(x_train_25, y_train_25, family = "binomial", alpha = opt_inputs(x_train_25, y_train_25)$alpha, lambda = opt_inputs(x_train_25, y_train_25)$lambda)

coef(elnet31_25)

selected_vars_25 <- c("Attr6", "Attr15", "Attr24", "Attr26", "Attr29", "Attr34", "Attr36", "Attr38", "Attr39", "Attr41", "Attr47")
selected_indices_25 <- match(selected_vars_25, colnames(x_train_25))
x_train_25_subset <- x_train_25[, selected_indices_25]
x_test_25_subset <- x_test_25[, selected_indices_25]

elnet11_25 <- glmnet(x_train_25_subset, y_train_25, family = "binomial", alpha = opt_inputs(x_train_25, y_train_25)$alpha, lambda = opt_inputs(x_train_25, y_train_25)$lambda)

# random forest

rf31_25 <- randomForest(data = train25, class ~ ., ntree = 500, importance = TRUE)
rf15_25 <- randomForest(data = train25, class ~ Attr15 + Attr41 + Attr9 + Attr25 + Attr40 + Attr6 + Attr48 + Attr33 + Attr29 + Attr3 + Attr39 + Attr36 + Attr23 + Attr13 + Attr26, ntree = 500, importance = TRUE)

# evaluating models

evaluation(logitstep_25, test25)
evaluation(logit_int_25, test25)
evaluation3(rf31_25, test25)
evaluation3(rf15_25, test25)

# comparing models using the ROC curve

ytest_25 <- as.vector(test25[, ncol(test25)])


predictions_rf15_25 <- predict(rf15_25, test25, type = "class")
predictions_logit15_25 <- predict(logit15_25, test25, type = "response")
predictions_logit15_25 <- ifelse(predictions_logit15_25 > 0.5, 1, 0)
predictions_logit_int_25 <- predict(logit_int_25, test25, type = "response")
predictions_logit_int_25 <- ifelse(predictions_logit_int_25 > 0.5, 1, 0)
predictions_elnet11_25 <- predict(elnet11_25, x_test_25_subset, type = "response")
predictions_elnet11_25 <- ifelse(predictions_elnet11_25 > 0.5, 1, 0)

num.ytest_25 <- as.numeric(ytest_25)


roc_rf15_25 <- roc(num.ytest_25, as.numeric(predictions_rf15_25))
roc_logit15_25 <- roc(as.vector(test25$class), predictions_logit15_25)
roc_logit_int_25 <- roc(as.vector(test25$class), predictions_logit_int_25)
roc_elnet11_25 <- roc(as.vector(test25$class), predictions_elnet11_25[, 1])


plot(roc_rf15_25, col = "olivedrab3", main = "ROC Curves for Multiple Models")
plot(roc_logit15_25, col = "violetred", add = TRUE)
plot(roc_logit_int_25, col = "hotpink4", add = TRUE)
plot(roc_elnet11_25, col = "cyan3", add = TRUE)

legend("bottomright", legend = c("Random Forest", "Logistic", "Logistic with Interaction", "Elastic Net"), 
       col = c("olivedrab3", "violetred", "hotpink4","cyan3"), lty = 1)


# makes the prediction much worse than the 50/50 dataset

# checking for overfitting

evaluation(logitstep_25, test25)
evaluation(logitstep_25, train25)

evaluation(logit_int_25, test25)
evaluation(logit_int_25, train25)

# not overfitted

evaluation3(rf15_25, test25)
evaluation3(rf15_25, train25)

# may be overfitted



##################################################


# simulation studies

# creating a bootstrap sample from the dataset: 50/50 split

balanced_bootstrap <- function(data, seed) {
  set.seed(seed)
  
  class1 <- subset(data, class == 1)
  class0 <- subset(data, class == 0)
  
  bootstrap1 <- class1[sample(nrow(class1), 500, replace = TRUE), ]
  
  bootstrap0 <- class0[sample(nrow(class0), 500, replace = TRUE), ]
  
  balanced_sample <- rbind(bootstrap1, bootstrap0)
  
  return(balanced_sample)
}

# creating training and testing datasets

split_data <- function(data, ratio) {
  n_train1 <- round(ratio * sum(data$class == 1))
  n_train0 <- round(ratio * sum(data$class == 0))
  
  indices1 <- sample(which(data$class == 1), n_train1)
  indices0 <- sample(which(data$class == 0), n_train0)
  
  train_data <- rbind(data[indices1, ], data[indices0, ])
  
  test_data <- data[-c(indices1, indices0), ]
  
  return(list(train_data = train_data, test_data = test_data))
}


seeds <- c(2365, 3461, 9387, 5396, 8648)

for (i in 1:5) {
  assign(paste0("sim", i), balanced_bootstrap(final_data, seeds[i]))
}

set.seed(416878)

for (i in 1:5) {
  assign(paste0("sim", i), split_data(get(paste0("sim", i)), 0.8))
}

# selecting same variables as 50/50

# running the models

for (i in 1:5) {
  train_data <- get(paste0("sim", i))
  
  log <- paste0("sim", i, "log")
  logint <- paste0("sim", i, "logint")
  rf <- paste0("sim", i, "rf")
  
  assign(log, glm(class ~ Attr29 + Attr61 + Attr53 + Attr28 + Attr13 + Attr23 + Attr48 + Attr3 + Attr25 + Attr6 + Attr34 + Attr39 + Attr36 + Attr40 + Attr26, data = train_data$train, family = binomial(link = "logit")))
  
  assign(logint, step(get(log), direction = "both", scope = class ~ (Attr29 + Attr61 + Attr53 + Attr28 + Attr13 + Attr23 + Attr48 + Attr3 + Attr25 + Attr6 + Attr34 + Attr39 + Attr36 + Attr40 + Attr26)^2))
  
  assign(rf, randomForest(data = train_data$train, class ~ Attr29 + Attr61 + Attr53 + Attr28 + Attr13 + Attr23 + Attr48 + Attr3 + Attr25 + Attr6 + Attr34 + Attr39 + Attr36 + Attr40 + Attr26, ntree = 500))
}

variables <- c("Attr29", "Attr61", "Attr53", "Attr28", "Attr13", "Attr23", "Attr48", "Attr3", "Attr25", "Attr6", "Attr34", "Attr39", "Attr36", "Attr40", "Attr26")

for (i in 1:5) {
  train_data <- get(paste0("sim", i))$train
  
  x.train <- as.matrix(train_data[, variables])
  y.train <- as.vector(train_data[, "class"])
  
  elnet_optimal <- opt_inputs(x.train, y.train)
  
  assign(paste0("sim", i, "elnet"), glmnet(x.train, y.train, family = "binomial", alpha = elnet_optimal$alpha, lambda = elnet_optimal$lambda))
}

# comparing models for each simulation using the ROC curve


library(pROC)


generate_roc <- function(rf_model, logit_int_model, logit_model, elnet_model, test_data) {
  predictions_rf <- predict(rf_model, test_data, type = "class")
  
  predictions_logit_int <- predict(logit_int_model, test_data, type = "response")
  predictions_logit_int <- ifelse(predictions_logit_int > 0.5, 1, 0)
  
  predictions_logit <- predict(logit_model, test_data, type = "response")
  predictions_logit <- ifelse(predictions_logit > 0.5, 1, 0)
  
  predictions_elnet <- predict(elnet_model, as.matrix(test_data[, variables]), type = "response")
  predictions_elnet <- ifelse(predictions_elnet > 0.5, 1, 0)
  
  ytest <- as.numeric(test_data[, ncol(test_data)])
  
  roc_rf <- roc(ytest, as.numeric(predictions_rf))
  roc_logit_int <- roc(ytest, as.numeric(predictions_logit_int))
  roc_logit <- roc(ytest, as.numeric(predictions_logit))
  roc_elnet <- roc(ytest, as.numeric(predictions_elnet))
  
  plot(roc_rf, col = "olivedrab3", main = "ROC Curves for Simulation Models")
  plot(roc_logit_int, col = "violetred", add = TRUE)
  plot(roc_logit, col = "hotpink4", add = TRUE)
  plot(roc_elnet, col = "cyan3", add = TRUE)
  
  legend("bottomright", legend = c("Random Forest", "Logistic with Interaction", "Logistic", "Elastic Net"),
         col = c("olivedrab3", "violetred", "hotpink4", "cyan3"), lty = 1)
}


# Generate ROC curves for each simulation

generate_roc(sim1rf, sim1logint, sim1log, sim1elnet, sim1$test)

generate_roc(sim2rf, sim2logint, sim2log, sim2elnet, sim2$test)

generate_roc(sim3rf, sim3logint, sim3log, sim3elnet, sim3$test)

generate_roc(sim4rf, sim4logint, sim4log, sim4elnet, sim4$test)

generate_roc(sim5rf, sim5logint, sim5log, sim5elnet, sim5$test)



