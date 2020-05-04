library(caret)
library(glmnet)

set.seed(7)

modelmetric <- data.frame(matrix(ncol = 4, nrow = 0))

housing <- read.csv('data/housing/Boston.csv', stringsAsFactors = F)
colnames(housing) <- c('idx', 'crim', 'zn', 'indus', 'chas', 'nox',
                    'rm', 'age', 'dis', 'rad', 'tax', 'ptratio',
                    'black', 'lstat', 'label')

#Drop idx
housing <- housing[, -1]
#auto <- auto[auto['cylinders']!=5,]
str(housing)

#Convert certain categorical columns to factors
housing$chas <- as.factor(housing$chas)

#Dummify the factor variables
dmy <- dummyVars("~.", data = housing)
df <- data.frame(predict(dmy, newdata = housing))

#Divide to train and test sets
inTraining <- createDataPartition(df$label, p = 0.75, list = F)
train <- df[inTraining, ]
test <- df[-inTraining, ]

##Linear Regression
linear_model <- train(
  label~., data = train, method = "glmnet",
  trControl = trainControl("repeatedcv", number = 3, repeats = 10),
  tuneGrid = expand.grid(lambda = c(0, 0.25, 0.5), alpha = c(0, 0.25, 0.5))
)
pred <- predict.train(object = linear_model, test)
error <- test$label - pred
modelmetric <- rbind(modelmetric, list("Linear Regression", cor(test$label, pred)^2, 
                                       sqrt(mean(error^2)), mean(abs(error))),
                     stringsAsFactors=FALSE)
colnames(modelmetric) <- c("model", "r2", "rmse", "mae")

##Decision Tree
decision_model <- train(
  label~., data = train, method = 'rpart2',
  trControl = trainControl("repeatedcv", number = 3, repeats = 10),
  tuneGrid = expand.grid(maxdepth = c(5, 10, 15))
)
pred <- predict.train(object = decision_model, test)
error <- test$label - pred
modelmetric <- rbind(modelmetric, list("Decision Tree", cor(test$label, pred)^2, 
                                       sqrt(mean(error^2)), mean(abs(error))),
                     stringsAsFactors=FALSE)

##Random Forest
store_maxtrees <- list()
for (n in c(10, 15, 20)){
  rf_model <- train(
    label~., data = train, method = 'rf',
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
  label~., data = train, method = 'rf',
  trControl = trainControl("repeatedcv", number = 3, repeats = 10),
  tuneGrid = expand.grid(mtry = round(ncol(train)/3)),
  ntree = 20
)
pred <- predict.train(object = rf_model, test)
error <- test$label - pred
modelmetric <- rbind(modelmetric, list("Random Forest", cor(test$label, pred)^2, 
                                       sqrt(mean(error^2)), mean(abs(error))),
                     stringsAsFactors=FALSE)

##Gradient Boosted Trees
getModelInfo("gbm")
gbm_model <- train(
  label~., data = train, method = 'gbm',
  trControl = trainControl("repeatedcv", number = 3, repeats = 10),
  distribution = 'gaussian'
  #tuneGrid = expand.grid(n.trees = c(10, 15, 20), interaction.depth = c(5, 10, 15),
  #shrinkage = seq(0, 2, by= 0.05), n.minobsinnode)
)
pred <- predict.train(object = gbm_model, test)
error <- test$label - pred
modelmetric <- rbind(modelmetric, list("Gradient Boosted Trees", cor(test$label, pred)^2, 
                                       sqrt(mean(error^2)), mean(abs(error))),
                     stringsAsFactors=FALSE)

write.csv(modelmetric,"r_pkg/metrics/housing_metric.csv",row.names=F,quote=F)
