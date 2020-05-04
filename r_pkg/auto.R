library(caret)
library(glmnet)

set.seed(7)

modelmetric <- data.frame(matrix(ncol = 4, nrow = 0),stringsAsFactors=FALSE)

auto <- read.csv('data/auto/auto-mpg.csv', stringsAsFactors = F)
colnames(auto) <- c('label', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration',
                    'model_year', 'origin', 'car_name')

#Drop cars with 3 or 5 cylinders
auto <- auto[auto['cylinders']!=3,]
auto <- auto[auto['cylinders']!=5,]

#Fill missing horsepower values with mean cylinders
auto$horsepower[is.na(auto$horsepower)] <- mean(auto$horsepower[!is.na(auto$horsepower)])

#Drop car names
auto <- auto[-9]

#Convert certain categorical columns to factors
auto$cylinders <- as.factor(auto$cylinders)
auto$model_year <- as.factor(auto$model_year)
auto$origin <- as.factor(auto$origin)

#Dummify the factor variables
dmy <- dummyVars("~.", data = auto)
df <- data.frame(predict(dmy, newdata = auto))

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
  ntree = 15
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
)
pred <- predict.train(object = gbm_model, test)
error <- test$label - pred
modelmetric <- rbind(modelmetric, list("Gradient Boosted Trees", cor(test$label, pred)^2, 
                                                            sqrt(mean(error^2)), mean(abs(error))),
                     stringsAsFactors=FALSE)

write.csv(modelmetric,"r_pkg/metrics/auto_metric.csv",row.names=F,quote=F)
