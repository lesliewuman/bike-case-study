# Bike Buyer Case Study

# Required packages
library(tidyverse)
library(broom)
library(corrplot)
library(cowplot)
library(GGally)
library(car)
library(ResourceSelection)
library(ROCR)
library(randomForest)
# library(gbm)
# library(caret)
theme_set(theme_classic())

set.seed(0) # Ensuring the results are reproducible

#### Reading in and cleaning data ####

bike <- read.table(file.choose(), header=TRUE, sep=",")
View(bike)

# Extra unwanted column in the data, Variable "X"
levels(bike$X)
which(bike$X=="No") # The problematic row(s)
bike[9,] # There is an unnecessary "NULL" in row 9
bike[9,2:14] <- type.convert(c(bike[9,3:14],""))
bike <- subset(bike, select=-c(X)) # Remove column "X" 

# Ensuring correct formatting of data
str(bike)
bike$Age <- as.numeric(levels(droplevels(bike$Age)))[bike$Age]
bike$Cars <- as.numeric(levels(droplevels(bike$Cars)))[bike$Cars]
bike$Income <- as.numeric(levels(droplevels(bike$Income)))[bike$Income]
bike$Income <- bike$Income/10000 # income in terms of ten thousands

# Converting NULL values to NA for improved handling in R
# and dealing with incorrect data value(s)
bike <- bike %>% replace(.=="NULL", NA)
bike$Cars[which(bike$Cars==-1)] <- NA # changing incorrect value

# Drop unused factor levels
bike <- droplevels(bike)

# Remove rows with missing values
bike <- na.omit(bike)

# ID variable unnecessary
bike <- bike[,-1]

str(bike)
summary(bike)

####

#### Exploratory Data Analysis ####

# Identifying numeric and factor variables
numeric.vars <- c("Income", "Cars", "Children", "Age")
factor.vars <- names(bike)[-match(numeric.vars, names(bike))]

# Barplots of factor variables
par(mfrow=c(2,4), mgp=c(1.5,1,0), mar=c(3,2,2,2))
myColors  <- c("mistyrose", "lavender")
attach(bike)
for(i in factor.vars){
  vals <- table(get(i))
  bp   <- barplot(vals, col=myColors, main=i, ylim=c(0,1.3*max(vals)))
  text(bp, vals+max(vals)/10, labels=vals, cex=1, pos=3, col="black") 
  text(bp, vals, labels=round(prop.table(vals),2), cex=1, pos=3, col="red") 
}
detach(bike)

# Distribution plots of numeric variables
numPlots <- lapply(numeric.vars, function(variable)
  ggplot(bike, aes(x=get(variable))) +
    theme_bw() +
    geom_histogram(bins=10, fill=myColors[1], color="black") +
    geom_vline(aes(xintercept=mean(get(variable))),
               color="red", linetype="dashed", size=1) +
    labs(y="No. of individuals", x="",
         title=variable) +
    theme(plot.title=element_text(hjust=0.5)))

plot_grid(plotlist=numPlots, ncol=length(numeric.vars))


# Correlation analysis with correlation matrix heat map
colorPalette <- colorRampPalette(c("royalblue3", "royalblue", "white", 
                                   "palevioletred", "palevioletred4"))(60)
par(mfrow=c(1,1), mar=c(5,4,4,2) + 0.1, mgp=c(3,1,0))
corrplot(cor(bike[, numeric.vars]), method=c("color"), col=colorPalette, outline=T, 
         tl.col="black", tl.srt=0, tl.offset=1, addCoef.col="black",
         number.cex=1, tl.cex=0.9)

####

#### Modelling ####

# Creating a training and testing set according to 80-20 heuristic
sampled.indices <- sample(1:nrow(bike), 0.8*nrow(bike))
train.set <- bike[sampled.indices, ]
test.set <- bike[-sampled.indices, ]

# Model 1 - logistic regression model
# Model building through forward selection with AIC as criterion
glm_constant <- glm(Purchased.Bike ~ 1, data=train.set, family="binomial")
glm_full <- glm(Purchased.Bike ~ ., data=train.set, family="binomial")
glm_final <- step(glm_constant, direction="forward", scope=list(lower=glm_constant, upper=glm_full))
summary(glm_final)
round(exp(glm_final$coefficients), 2) # odds ratios
round(exp(confint(glm_final)), 2) # 0.95 confidence interval of odds ratios
ggcoef(glm_final, conf.int=TRUE, exponentiate=TRUE, sort="ascending", mapping=aes(x=estimate, y=term, size=p.value),
       vline_color="red", vline_linetype= "solid", errorbar_color="blue", errorbar_height=.35)

# Model diagnostics 
# Hosmer-Lemeshow test for model goodness-of-fit
# Research suggests that g > p+1 where p is the number of indep. variables in the final model
(hl <- hoslem.test(ifelse(train.set$Purchased.Bike=="No", 0, 1), fitted(glm_final), g=10))
# Large p-value -- cannot reject hypothesis of good fit
hl_df <- data.frame(obs_p=hl$observed[,2], exp_p=hl$expected[,2])
ggplot(hl_df, aes(x=obs_p, y=exp_p)) +
  geom_point() +
  geom_smooth() +
  geom_abline(intercept=0, slope=1, size=0.5)  # reference line

# Linearity assumption
# Bind the logit and tidying the data for plot
probabilities <- predict(glm_final, type="response")
df <- train.set[, c("Cars","Children","Income")] %>%
  mutate(logit=log(probabilities/(1-probabilities))) %>%
  gather(key="predictors", value="predictor.value", -logit)
ggplot(df, aes(logit, predictor.value))+
  geom_point(size=0.5, alpha=0.5) +
  geom_smooth(method="loess") + 
  theme_bw() + 
  facet_wrap(~predictors, scales="free_y")

# Multicollinearity assessment
vif(glm_final)

# Classification scheme - determining threshold
glm_predictionObject <- prediction(fitted(glm_final), labels=train.set$Purchased.Bike)
roc <- performance(glm_predictionObject,"tpr","fpr")
AUC <- round(as.numeric(performance(glm_predictionObject,"auc")@y.values),4)
# determination of threshold using Youden's J statistic
cutoffs <- data.frame(cut=roc@alpha.values[[1]], tpr=roc@y.values[[1]], spec=1 - roc@x.values[[1]],
                      fpr=roc@x.values[[1]])
J.index <- which.max(apply(cutoffs[,c(2,3)],1,function(x) x[1]+x[2]-1))
threshold <- cutoffs[J.index,1]

plot(roc, lwd=3, colorize=TRUE)
abline(a=0, b=1, col="black", lwd=1.5, lty="dashed")
text(0.4,0.55, labels=paste("AUC =", AUC))
points(cutoffs[J.index,4], cutoffs[J.index,2], col="black", lwd=2)
text(cutoffs[J.index,4]-0.02, cutoffs[J.index,2]+0.15, "Optimal cut-off")
text(cutoffs[J.index,4]-0.02, cutoffs[J.index,2]+0.075, paste("Threshold", round(threshold,2)))

# # Performance on training data
# pred_train <- predict(glm_final, type="response")
# pred_train <- factor(ifelse(pred_train > threshold, 1, 0), levels=c(0, 1), labels=c("No", "Yes"))
# (confMatrix_train <- table(pred_train, train.set$Purchased.Bike, dnn=c("Pred","Obs")))
# 
# (sens_train <- confMatrix_train[2,2]/(confMatrix_train[2,2]+confMatrix_train[1,2]))
# (spec_train <- confMatrix_train[1,1]/(confMatrix_train[1,1]+confMatrix_train[2,1]))

# Validation - comparing classification scheme to out-of-sample data
pred_glm <- predict(glm_final, newdata=test.set, type="response")
pred_glm <- factor(ifelse(pred_glm>threshold, 1, 0), levels=c(0, 1), labels=c("No", "Yes"))
(confMatrix_glm <- table(pred_glm, test.set$Purchased.Bike, dnn=c("Pred","Obs")))

(sens_glm <- confMatrix_glm[2,2]/(confMatrix_glm[2,2]+confMatrix_glm[1,2]))
(spec_glm <- confMatrix_glm[1,1]/(confMatrix_glm[1,1]+confMatrix_glm[2,1]))
(misclassError_glm <- 1 - sum(diag(confMatrix_glm))/sum(confMatrix_glm))

# Model 2 - bagging and random forest
# Fit a bagged trees model
bag <- randomForest(Purchased.Bike ~ ., data=train.set,
                    mtry=ncol(train.set)-1, # for bagging, use all predictors
                    ntree=10000, 
                    importance=TRUE, # keep track of reduction in loss function
                    do.trace=1000) 
bag

# Variable importance plot
par(mar=c(4,10,2,2))
varimp_bag <- importance(bag, type=2) # type=2: Reduction in gini index
varimp_bag <- varimp_bag[order(varimp_bag, decreasing=FALSE),]
barplot(varimp_bag, horiz=T, col="navy", las=1,
        xlab="Mean decrease in Gini index", cex.lab=1, cex.axis=1, 
        main="Variable Importance of Bagged Model", cex.main=1.1, cex.names=1)

# Fit a random forest model
rf <- randomForest(Purchased.Bike ~ ., data=train.set,
                   ntree=10000, 
                   importance=TRUE, 
                   do.trace=1000) 
rf

# Variable importance plot
par(mar=c(4,10,2,2))
varimp_rf <- importance(rf, type=2) # type=2: Reduction in gini index
varimp_rf <- varimp_rf[order(varimp_rf, decreasing=FALSE),]
barplot(varimp_rf, horiz=T, col="navy", las=1,
        xlab="Mean decrease in Gini index", cex.lab=1, cex.axis=1, 
        main="Variable Importance of RF Model", cex.main=1.1, cex.names=1)

# compare OOB errors:
par(mar=c(5,4,4,2) + 0.1)
plot(rf$err.rate[, "OOB"], type="s", xlab="Number of trees", ylab="OOB error")
lines(bag$err.rate[, "OOB"], col="navy", type="s")
legend("topright", legend=c("Bagged", "Random Forest"), 
       col=c("navy", "black"), lwd=2, bty="n")

# the relationship of age with purchased.bike
plot(bike$Age, as.numeric(bike$Purchased.Bike)-1, yaxt='n',
     xlab="Age", ylab="Purchased.Bike", ylim=c(-0.5,1.5))
axis(2, at=c(0,1), labels=c("No", "Yes"), las=1)

# Prediction - comparing classification scheme to out-of-sample data
pred_bag <- predict(bag, newdata=test.set)
pred_rf <- predict(rf, newdata=test.set) 

(confMatrix_bag <- table(pred_bag, test.set$Purchased.Bike, dnn=c("Pred", "Obs")))
(confMatrix_rf <- table(pred_rf, test.set$Purchased.Bike, dnn=c("Pred", "Obs")))

(sens_bag <- confMatrix_bag[2,2]/(confMatrix_bag[2,2]+confMatrix_bag[1,2]))
(spec_bag <- confMatrix_bag[1,1]/(confMatrix_bag[1,1]+confMatrix_bag[2,1]))
(sens_rf <- confMatrix_rf[2,2]/(confMatrix_rf[2,2]+confMatrix_rf[1,2]))
(spec_rf <- confMatrix_rf[1,1]/(confMatrix_rf[1,1]+confMatrix_rf[2,1]))

(misclassError_bag <- mean(pred_bag != test.set$Purchased.Bike))
(misclassError_rf <- mean(pred_rf != test.set$Purchased.Bike))

# # Model 3 - gradient boosting model 
# # (doesn't yield significant improvement over model 2)
# 
# # GBM method
# # Hyperparameters determined via a grid research
# ctrl <- trainControl(method="cv", number=20, verboseIter=T)
# gbm_grid <- expand.grid(n.trees=c(2000, 5000, 10000),
#                         interaction.depth=c(1, 2, 5, 10),
#                         shrinkage=c(0.1, 0.01),
#                         n.minobsinnode=1)
# 
# 
# gbm_gridsearch <- train(Purchased.Bike ~ ., data=train.set, 
#                         method="gbm", 
#                         distribution="bernoulli", 
#                         trControl=ctrl, 
#                         verbose=F, 
#                         tuneGrid=gbm_grid)
# gbm_gridsearch
# 
# pred_gbm <- predict(gbm_gridsearch, test.set)
# (confMatrix_gbm <- table(pred_gbm, test.set$Purchased.Bike, dnn=c("Pred","Obs")))
# 
# (sens_gbm <- confMatrix_gbm[2,2]/(confMatrix_gbm[2,2]+confMatrix_gbm[1,2]))
# (spec_gbm <- confMatrix_gbm[1,1]/(confMatrix_gbm[1,1]+confMatrix_gbm[2,1]))
# (misclassError_glm <- 1 - sum(diag(confMatrix_gbm))/sum(confMatrix_gbm))
# 
# # XGBoost method
# # Hyperparameters determined via a grid research
# ctrl <-  trainControl(method="cv", number=20, verboseIter=T)
# xgb_grid <- expand.grid(nrounds = c(2000, 5000, 10000),  # number of trees
#                         max_depth = c(1, 2, 5, 10),      # interaction depth
#                         eta = c(0.1, 0.01),              # learning rate
#                         gamma = 0.001,                   # mindev
#                         colsample_bytree = 1,            # proportion random features per tree
#                         min_child_weight = 1,            # also controls tree depth
#                         subsample = c(0.5,1))            # bootstrap proportion
# 
# xgb_gridsearch <- train(Purchased.Bike ~ ., data=train.set, 
#              method="xgbTree",
#              trControl=ctrl,
#              verbose= F,
#              tuneGrid=xgb_grid)
# xgb_gridsearch
# xgb_gridsearch$bestTune
# 
# pred_xgb <- predict(xgb_gridsearch, test.set)
# (confMatrix_xgb <- table(pred_xgb, test.set$Purchased.Bike, dnn=c("Pred","Obs")))
# (sens_xgb <- confMatrix_xgb[2,2]/(confMatrix_xgb[2,2]+confMatrix_xgb[1,2]))
# (spec_xgb <- confMatrix_xgb[1,1]/(confMatrix_xgb[1,1]+confMatrix_xgb[2,1]))
# (misclassError_xgb <- 1 - sum(diag(confMatrix_xgb))/sum(confMatrix_xgb))

####


