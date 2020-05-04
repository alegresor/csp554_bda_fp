library(caret)
library(glmnet)

set.seed(7)

modelmetric <- data.frame(matrix(ncol = 4, nrow = 0),stringsAsFactors=FALSE)

red <- read.csv('data/wine/winequality-red.csv', stringsAsFactors = F, sep = ";")
white <- read.csv('data/wine/winequality-white.csv', stringsAsFactors = F, sep = ";")

#Add color classification
red['color'] <- 'red'
white['color'] <- 'white'

merged <- rbind(red, white)

#Convert certain categorical columns to factors
merged$color <- as.factor(merged$color)

#Dummify the factor variables
dmy <- dummyVars("~.", data = merged)
df <- data.frame(predict(dmy, newdata = merged))

#Divide to train and test sets
inTraining <- createDataPartition(df$quality, p = 0.75, list = F)
train <- df[inTraining, ]
test <- df[-inTraining, ]

##Linear Regression
linear_model <- train(
  quality~., data = train, method = "glmnet",
  trControl = trainControl("repeatedcv", number = 3, repeats = 10),
  tuneGrid = expand.grid(lambda = c(0, 0.25, 0.5), alpha = c(0, 0.25, 0.5))
)
pred <- predict.train(object = linear_model, test)
error <- test$quality - pred
modelmetric <- rbind(modelmetric, list("Linear Regression", cor(test$quality, pred)^2, 
                                       sqrt(mean(error^2)), mean(abs(error))),
                     stringsAsFactors=FALSE)
colnames(modelmetric) <- c("model", "r2", "rmse", "mae")

##Decision Tree
decision_model <- train(
  quality~., data = train, method = 'rpart2',
  trControl = trainControl("repeatedcv", number = 3, repeats = 10),
  tuneGrid = expand.grid(maxdepth = c(5, 10, 15))
)
pred <- predict.train(object = decision_model, test)
error <- test$quality - pred
modelmetric <- rbind(modelmetric, list("Decision Tree", cor(test$quality, pred)^2, 
                                       sqrt(mean(error^2)), mean(abs(error))),
                     stringsAsFactors=FALSE)

##Random Forest
store_maxtrees <- list()
for (n in c(10, 15, 20)){
  rf_model <- train(
    quality~., data = train, method = 'rf',
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
  quality~., data = train, method = 'rf',
  trControl = trainControl("repeatedcv", number = 3, repeats = 10),
  tuneGrid = expand.grid(mtry = round(ncol(train)/3)),
  ntree = 20
)
pred <- predict.train(object = rf_model, test)
error <- test$quality - pred
modelmetric <- rbind(modelmetric, list("Random Forest", cor(test$quality, pred)^2, 
                                       sqrt(mean(error^2)), mean(abs(error))),
                     stringsAsFactors=FALSE)

##Gradient Boosted Trees
gbm_model <- train(
  quality~., data = train, method = 'gbm',
  trControl = trainControl("repeatedcv", number = 3, repeats = 10),
  distribution = 'gaussian'
)
pred <- predict.train(object = gbm_model, test)
error <- test$quality - pred
modelmetric <- rbind(modelmetric, list("Gradient Boosted Trees", cor(test$quality, pred)^2, 
                                       sqrt(mean(error^2)), mean(abs(error))),
                     stringsAsFactors=FALSE)

write.csv(modelmetric,"r_pkg/metrics/wine.csv",row.names=F,quote=F)

