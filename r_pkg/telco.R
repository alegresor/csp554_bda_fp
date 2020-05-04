library(caret)
library(glmnet)
library(e1071)
library(mltest)

set.seed(7)

modelmetric <- data.frame(matrix(ncol = 4, nrow = 0))

telco <- read.csv('data/telco/WA_Fn-UseC_-Telco-Customer-Churn.csv', stringsAsFactors = F, header = T)

#Data cleaning
telco[,'months'] <- telco$TotalCharges / telco$MonthlyCharges
telco <- telco[, -c(1, 20)]
telco <- telco[complete.cases(telco),]

#Convert certain categorical columns to factors
num <- telco[,c(5, 18, 20)]
cat <- telco[,-c(5, 18, 20)]
cat[] <- lapply(cat, function(y) as.factor(y))

#Dummify the factor variables
dmy <- dummyVars("~.", data = cat[,-17])
df <- data.frame(predict(dmy, newdata = cat[,-17]))
df <- cbind(cat[,17], num, df)
str(df)
colnames(df)[1] <- "churn"

#Divide to train and test sets
inTraining <- createDataPartition(df$churn, p = 0.75, list = F)
train <- df[inTraining, ]
test <- df[inTraining, ]

#Weighted Recall Function
weightedRecall <- function(met, conf){
  added <- 0
  for (i in 1:length(met$recall)){
    added <- added + (metric$recall[[i]] * sum(confusion$table[,i]))
  }
  return (added/nrow(test))
}

#Weighted Precision Function
weightedPrecision <- function(met, conf){
  added <- 0
  for (i in 1:length(met$precision)){
    added <- added + (metric$precision[[i]] * sum(confusion$table[,i]))
  }
  return (added/nrow(test))
}

##Linear Regression
log_model <- train(
  churn~., data = train, method = "glmnet",
  trControl = trainControl("repeatedcv", number = 3, repeats = 10),
  tuneGrid = expand.grid(lambda = c(0, 0.25, 0.5), alpha = c(0, 0.25, 0.5))
)
pred <- predict.train(object = log_model, test)
metric <- ml_test(pred, test$churn)
confusion <- confusionMatrix(pred, test$churn)
modelmetric <- rbind(modelmetric, list("Linear Regression", metric$accuracy, weightedRecall(metric, confusion),
                                       weightedPrecision(metric, confusion)),
                     stringsAsFactors=FALSE)
colnames(modelmetric) <- c("model", "accuracy", "weightedRecall", "weightedPrecision")

##Decision Tree
decision_model <- train(
  churn~., data = train, method = 'rpart2',
  trControl = trainControl("repeatedcv", number = 3, repeats = 10),
  tuneGrid = expand.grid(maxdepth = c(5, 10, 15))
)
pred <- predict.train(object = decision_model, test)
metric <- ml_test(pred, test$churn)
confusion <- confusionMatrix(pred, test$churn)
modelmetric <- rbind(modelmetric, list("Decision Tree", metric$accuracy, weightedRecall(metric, confusion),
                                       weightedPrecision(metric, confusion)),
                     stringsAsFactors=FALSE)

##Random Forest
store_maxtrees <- list()
for (n in c(10, 15, 20)){
  rf_model <- train(
    churn~., data = train, method = 'rf',
    trControl = trainControl("repeatedcv", number = 3, repeats = 10),
    tuneGrid = expand.grid(mtry = round(ncol(train)/3)),
    ntree = n
  )
  key <- toString(n)
  store_maxtrees[[key]] <- rf_model
}
results <- resamples(store_maxtrees)
summary(results)

rf_model <- train(
  churn~., data = train, method = 'rf',
  trControl = trainControl("repeatedcv", number = 3, repeats = 10),
  tuneGrid = expand.grid(mtry = round(ncol(train)/3)),
  ntree = 20
)
pred <- predict.train(object = rf_model, test)
metric <- ml_test(pred, test$species)
confusion <- confusionMatrix(pred, test$species)
modelmetric <- rbind(modelmetric, list("Random Forest", metric$accuracy, weightedRecall(metric, confusion),
                                       weightedPrecision(metric, confusion)),
                     stringsAsFactors=FALSE)

##Naive Bayes
nb_model <- train(
  churn~., data = train, method = 'nb',
  trControl = trainControl("repeatedcv", number = 3, repeats = 10),
  tuneGrid = expand.grid(fL = c(0.5, 1, 2), usekernel = T, adjust = c(0.5, 1, 2))
)
pred <- predict.train(object = nb_model, test)
metric <- ml_test(pred, test$churn)
confusion <- confusionMatrix(pred, test$churn)
modelmetric <- rbind(modelmetric, list("Naive Bayes", metric$accuracy, weightedRecall(metric, confusion),
                                       weightedPrecision(metric, confusion)),
                     stringsAsFactors=FALSE)


##Gradient Boosted Trees
gbm_model <- train(
  churn~., data = train, method = 'gbm',
  trControl = trainControl("repeatedcv", number = 3, repeats = 10),
  tuneGrid = expand.grid(n.trees = c(10, 15, 20), interaction.depth = c(5, 10, 15),
                         shrinkage = 0.1, n.minobsinnode = 10)
)
pred <- predict.train(object = gbm_model, test)
metric <- ml_test(pred, test$churn)
confusion <- confusionMatrix(pred, test$churn)
metric$accuracy
modelmetric <- rbind(modelmetric, list("Gradient Boosted Trees", metric$accuracy, weightedRecall(metric, confusion),
                                       weightedPrecision(metric, confusion)),
                     stringsAsFactors=FALSE)

##Linear SVM
svm_model <- train(
  churn~., data = train, method = 'svmLinear',
  trControl = trainControl("repeatedcv", number = 3, repeats = 10)
)
pred <- predict.train(object = svm_model, test)
metric <- ml_test(pred, test$churn)
confusion <- confusionMatrix(pred, test$churn)
modelmetric <- rbind(modelmetric, list("Linear SVM", metric$accuracy, weightedRecall(metric, confusion),
                                       weightedPrecision(metric, confusion)),
                     stringsAsFactors=FALSE)

write.csv(modelmetric,"r_pkg/metrics/telco_metric.csv",row.names=F,quote=F)

