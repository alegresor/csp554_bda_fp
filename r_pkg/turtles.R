library(caret)
library(glmnet)
library(e1071)
library(mltest)

set.seed(7)

modelmetric <- data.frame(matrix(ncol = 4, nrow = 0))

turtle <- read.csv('data/turtles/turtles_clean_pandas.csv', stringsAsFactors = F)

colnames(turtle) <- c('species', 'dead_alive', 'gear', 'scl_notch', 'scl_tip',
                      'scw', 'ccl_notch', 'testLevel_before', 'entangled')

#Convert certain categorical columns to factors

turtle$dead_alive <- as.factor(turtle$dead_alive)
turtle$gear <- as.factor(turtle$gear)
turtle$entangled <- as.factor(turtle$entangled)

#Dummify the factor variables
dmy <- dummyVars("species~.", data = turtle)
df <- data.frame(predict(dmy, newdata = turtle))
df <- cbind(turtle$species, df)
colnames(df)[1] <- "species"
df$species <- as.factor(df$species)

#Divide to train and test sets
inTraining <- createDataPartition(df$species, p = 0.75, list = F)
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
  species~., data = train, method = "glmnet",
  trControl = trainControl("repeatedcv", number = 3, repeats = 10),
  tuneGrid = expand.grid(lambda = c(0, 0.25, 0.5), alpha = c(0, 0.25, 0.5))
)
pred <- predict.train(object = log_model, test)
metric <- ml_test(pred, test$species)
confusion <- confusionMatrix(pred, test$species)
modelmetric <- rbind(modelmetric, list("Linear Regression", metric$accuracy, weightedRecall(metric, confusion),
                                       weightedPrecision(metric, confusion)),
                     stringsAsFactors=FALSE)
colnames(modelmetric) <- c("model", "accuracy", "weightedRecall", "weightedPrecision")

##Decision Tree
decision_model <- train(
  species~., data = train, method = 'rpart2',
  trControl = trainControl("repeatedcv", number = 3, repeats = 10),
  tuneGrid = expand.grid(maxdepth = c(5, 10, 15))
)
pred <- predict.train(object = decision_model, test)
metric <- ml_test(pred, test$species)
confusion <- confusionMatrix(pred, test$species)
modelmetric <- rbind(modelmetric, list("Decision Tree", metric$accuracy, weightedRecall(metric, confusion),
                                       weightedPrecision(metric, confusion)),
                     stringsAsFactors=FALSE)

##Random Forest
store_maxtrees <- list()
for (n in c(10, 15, 20)){
  rf_model <- train(
    species~., data = train, method = 'rf',
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
  species~., data = train, method = 'rf',
  trControl = trainControl("repeatedcv", number = 3, repeats = 10),
  tuneGrid = expand.grid(mtry = round(ncol(train)/3)),
  ntree = 15
)
pred <- predict.train(object = rf_model, test)
metric <- ml_test(pred, test$species)
confusion <- confusionMatrix(pred, test$species)
modelmetric <- rbind(modelmetric, list("Random Forest", metric$accuracy, weightedRecall(metric, confusion),
                                       weightedPrecision(metric, confusion)),
                     stringsAsFactors=FALSE)

##Naive Bayes
nb_model <- train(
  species~., data = train, method = 'nb',
  trControl = trainControl("repeatedcv", number = 3, repeats = 10),
  tuneGrid = expand.grid(fL = c(0.5, 1, 2), usekernel = T, adjust = c(0.5, 1, 2))
)
pred <- predict.train(object = nb_model, test)
metric <- ml_test(pred, test$species)
confusion <- confusionMatrix(pred, test$species)
modelmetric <- rbind(modelmetric, list("Naive Bayes", metric$accuracy, weightedRecall(metric, confusion),
                                       weightedPrecision(metric, confusion)),
                     stringsAsFactors=FALSE)

write.csv(modelmetric,"r_pkg/metrics/turtle_metric.csv",row.names=F,quote=F)
