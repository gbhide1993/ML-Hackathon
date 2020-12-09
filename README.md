# ML-Hackathon
Deal Id selection 
---
title: "Deal Id Selection Documentation"
author: "Rajesh Ajjarapu & Girish Bhide"
date: "12/9/2020"
output: html_document
---
The objective of this project is to classify the deal ids which has the highest chances of getting selected by the DSP. Also, to create a list of top 50 deal ids which can be passed directly to Ad Server. 



First, let's load the required packages for the code.

```{r, message=FALSE, warning=FALSE}
library(caret)
library(rpart)
library(dplyr)
library(randomForest)
library(standardize)
library(caTools)
library(tidyr)
library(readxl)
library(stringr)
library(jsonlite)
library(purrr)
library(ROCR)
library(e1071)
library(xgboost)
```

Setting up the working directory. You can change the directory as per your file location.

```{r}
setwd("C:/Users/test/Desktop/ML Hackathon")
```

Now, let's read the data from our directory. This data file is in Excel hence using (.xlsx) extension after file name. Change the extension as per your file format. After reading the data set will take a look at first 6 entries from the data set.

```{r}
df <- read_excel("Deal data TET-7506.xlsx")
head(df)
```

The data set has 10 variables out of which our dependent variable for this project is "c.is_winner". The independent variable "mbr" needs to be cleaned first and transform into clean data frame. Also, we will drop those variables which does not affect our selection process and add variables for price differences.

```{r, warning=FALSE}
mbr <- data.frame(mbr = df$mbr)

mbr <- map_dfr(df$mbr,fromJSON) # ! This takes around 10 mins !

df <- cbind(mbr,df)

df$price_diff_fp <- df$fp - df$pp
df$price_diff_sp <- df$sp - df$pp
```

Let's clean the data for NULL and missing values. Prepare data set on which we can process further analysis.

```{r, warning=FALSE}
df <- select(df, c("pid", "cmpid", "dsp_id", "deal_id", "wdid",
                   "wiabcid", "fp", "sp", "pp", "mpe", "spm",
                   "shp", "shb","price_diff_fp", "price_diff_sp", "c.is_winner"))

df$deal_id <- as.numeric(df$deal_id)                   
df$deal_id[is.na(df$deal_id)] <- 0
df$wdid[is.na(df$wdid)] <- 0    
df_1 <- separate(df, col = wiabcid, into = c("IAB1", "IAB2"), sep = ",") 
df_1 <- separate(df_1, col = IAB1, into = c("IAB_class","remove"), sep = "-")
df_1 <- subset(df_1, select = -c(IAB2, remove))

df_1$IAB_class <- as.factor(df_1$IAB_class) 
```

Now, let's split the data set into training and testing data sets. 70% data is kept for training and remaining 30% is for testing the model.

```{r, warning=FALSE}
split_df <- sample.split(df_1$c.is_winner, SplitRatio = 0.7)
train_df <- subset(df_1, split_df == TRUE)
test_df <- subset(df_1,split_df == FALSE)

train_df <-  na.omit(train_df)
test_df <- na.omit(test_df)
```

Let's scale the variables 

```{r, warning=FALSE, message=FALSE}
df_1$mpe <- scale(df_1$mpe)
df_1$spm <- scale(df_1$spm)
df_1$shp <- scale(df_1$shp)
df_1$shb <- scale(df_1$shb)
df_1$price_diff_fp <- scale(df_1$price_diff_fp)
df_1$price_diff_sp <- scale(df_1$price_diff_sp)
```

Selecting the model is important task to do. As we need to classify our deal ids into whether they will win or loose, we will use gradient boosting algorithm (XGBoost) which is good at classification tasks. Other classification models such as logistic regression and Support Vector Machines (SVM) were also used.

If we take a look at the ratio of winning to loosing deals we can clearly identify that this is the case of class imbalance. So, using parameter like accuracy to compare the models will be misleading.

Hence, the Receiver operating characteristic or ROC AUC is used to compare models.

First let's take a look at performance and plot of ROC AUC for logistic regression.

```{r, echo=FALSE}

model_glm <- glm(c.is_winner~ IAB_class + mpe + spm + shp + shb +
                price_diff_fp + price_diff_sp , 
              data = train_df, family = "binomial"(link = "logit"), maxit=100)# maxit added for convergence
# Prediction on test data set #
pred_val_glm <- predict(model_glm, test_df, type = "response") 
# Confusion Matrix for GLM#
#cfm <- table(test_df$c.is_winner, pred_val_glm>0.5)
#cfm
pred_val_glm <- ifelse(pred_val_glm>0.5,1,0)
confusionMatrix(data = as.factor(pred_val_glm), reference = as.factor(test_df$c.is_winner))
# ROC AUC for logistic regression #
pred_glm <- prediction(pred_val_glm, test_df$c.is_winner)
roc_curve_glm <- performance(pred_glm, "tpr", "fpr")
plot(roc_curve_glm,main= "Logistic Regression ROC Curve", col.main= "blue",
     col.lab= "blue")
```
Now, as we can see there are lots of falsely predicted values. 
Let's check how SVM performs on the same data set.

```{r, echo=FALSE}
model_svm <- svm(c.is_winner~ IAB_class + mpe + spm + shp + shb +      # <- !! This takes around 10 mins !!
                   price_diff_fp + price_diff_sp , 
                 data = train_df, type = "C-classification", kernel= "linear")
svm_pred_val <- predict(model_svm,test_df, type= "response")

# Confusion Matrix for SVM #
confusionMatrix(data = as.factor(svm_pred_val), reference = as.factor(test_df$c.is_winner))

# ROC AUC for SVM #
pred_svm <- prediction(as.numeric(svm_pred_val), test_df$c.is_winner)
roc_curve_svm <- performance(pred_svm, "tpr", "fpr")
plot(roc_curve_svm,main= "SVM ROC Curve", col.main= "blue")
```
We can see that both models performed almost same. We need a better classifier to complete our task. Let's check the performance of XGBoost.

```{r, warning=FALSE, message=FALSE}
model_xgb <- xgboost(data = data.matrix(train_df), label = train_df$c.is_winner,
                     eta = 0.1, # learning rate , lower val more robust to overfitting
                     max_depth = 15, 
                     nround=25, 
                     subsample = 0.5,
                     colsample_bytree = 0.5,
                     seed = 1,
                     eval_metric = "merror",
                     objective = "multi:softprob",
                     num_class = 2,
                     nthread = 3)

xgb_pred_val <- predict(model_xgb, data.matrix(test_df), type= "response")
xgb_pred_revised <- matrix(xgb_pred_val, nrow = 2, ncol = length(xgb_pred_val)/2) %>%
  t() %>% 
  data.frame() %>%
  mutate(label= test_df$c.is_winner, max_prob= max.col(., "last")-1)
logs <- model_xgb$evaluation_log
plot(logs$iter, logs$train_merror, col= "blue")

# Confusion Matrix for XGBoost #
xgb_pred_val <- ifelse(xgb_pred_val>0.5,1,0)
confusionMatrix(data = as.factor(xgb_pred_revised$max_prob), reference = as.factor(test_df$c.is_winner))

# ROC AUC for XGBoost #
pred_xgb <- prediction(as.numeric(xgb_pred_revised$max_prob), test_df$c.is_winner)
roc_curve_xgb <- performance(pred_xgb, "tpr", "fpr")
plot(roc_curve_xgb, main= "XGBoost ROC Curve", col.main= "blue")
```

We can clearly see that gradient boosting outperformed other two models. 
Now, let's move to main objective of this code which generating a simple list of top 50 deal ids which has the highest probability of getting selected by the DSP.

```{r, warning=FALSE, message=FALSE, echo=FALSE}
select_df <- data.frame(pub_id= test_df$pid, campaign_id= test_df$cmpid,
                        deal_id= test_df$deal_id, selection= xgb_pred_revised$max_prob)

deal_id_suceess <- table(deal_id= select_df$deal_id, selected= select_df$selection) %>% data.frame()
deal_id_suceess <- deal_id_suceess[!(deal_id_suceess$selected == 0),]
deal_id_suceess <- deal_id_suceess[!(deal_id_suceess$deal_id == 0),]
deal_id_suceess <- deal_id_suceess[order(deal_id_suceess$Freq, decreasing = TRUE),]
deal_list <- print(head(deal_id_suceess$deal_id,50))
head(deal_list)
deal_list
```

To test the model how it will perform on a completely unknown data set, we will run the model on a data set which was fetched from different data and time and for different time frame. 

```{r,warning=FALSE, message=FALSE, echo=FALSE}
df_valid <- read_excel("deal_data_2.xlsx")

mbr_valid <- data.frame(mbr_valid = df_valid$mbr)

mbr_valid <- map_dfr(df_valid$mbr, fromJSON)

df_valid <- cbind(mbr_valid,df_valid)

df_valid$price_diff_fp <- df_valid$fp - df_valid$pp
df_valid$price_diff_sp <- df_valid$sp - df_valid$pp

df_valid <- select(df_valid, c("pid", "cmpid", "dsp_id", "deal_id", "wdid",
                   "wiabcid", "fp", "sp", "pp", "mpe", "spm",
                   "shp", "shb","price_diff_fp", "price_diff_sp", "c.is_winner"))
df_valid$deal_id <- as.numeric(df_valid$deal_id)                   
df_valid$deal_id[is.na(df_valid$deal_id)] <- 0
df_valid$wdid[is.na(df_valid$wdid)] <- 0    
df_valid_1 <- separate(df_valid, col = wiabcid, into = c("IAB1", "IAB2"), sep = ",") 
df_valid_1 <- separate(df_valid_1, col = IAB1, into = c("IAB_class","remove"), sep = "-")
df_valid_1 <- subset(df_valid_1, select = -c(IAB2, remove))

df_valid_1$IAB_class <- as.factor(df_valid_1$IAB_class) 
df_valid_1 <- na.omit(df_valid_1)

### Scaling Validation data set ###
df_valid_1$mpe <- scale(df_valid_1$mpe)
df_valid_1$spm <- scale(df_valid_1$spm)
df_valid_1$shp <- scale(df_valid_1$shp)
df_valid_1$shb <- scale(df_valid_1$shb)
df_valid_1$price_diff_fp <- scale(df_valid_1$price_diff_fp)
df_valid_1$price_diff_sp <- scale(df_valid_1$price_diff_sp)

### Predicting on Validation Dataset ###

xgb_valid_pred <- predict(model_xgb, data.matrix(df_valid_1), type= "response")
xgb_valid_pred_revised <- matrix(xgb_valid_pred, nrow = 2, ncol = length(xgb_valid_pred)/2) %>%
  t() %>% 
  data.frame() %>%
  mutate(label= df_valid_1$c.is_winner , max_prob= max.col(., "last")-1)
#logs <- model_xgb$evaluation_log
#plot(logs$iter, logs$train_merror, col= "blue")

# Confusion Matrix for XGBoost on Validation Dataset #
xgb_valid_pred <- ifelse(xgb_valid_pred>0.5,1,0)
confusionMatrix(data = as.factor(xgb_valid_pred_revised$max_prob), reference = as.factor(df_valid_1$c.is_winner))

# ROC AUC for XGBoost #
pred_xgb_valid <- prediction(as.numeric(xgb_valid_pred_revised$max_prob), df_valid_1$c.is_winner)
roc_curve_xgb_valid <- performance(pred_xgb_valid, "tpr", "fpr")
plot(roc_curve_xgb_valid, main= "XGBoost ROC Curve for Validation Data", col.main= "blue")

# Selecting top 50 deal ids #

select_df_valid <- data.frame(pub_id= df_valid_1$pid, campaign_id= df_valid_1$cmpid,
                        deal_id= df_valid_1$deal_id, selection= xgb_valid_pred_revised$max_prob)

deal_id_success_valid <- table(deal_id= select_df_valid$deal_id, selected= select_df_valid$selection) %>% data.frame()
deal_id_success_valid <- deal_id_success_valid[!(deal_id_success_valid$selected == 0),]
deal_id_success_valid <- deal_id_success_valid[!(deal_id_success_valid$deal_id == 0),]
deal_id_success_valid <- deal_id_success_valid[order(deal_id_success_valid$Freq, decreasing = TRUE),]
deal_list_valid <- print(head(deal_id_success_valid$deal_id,50))
head(deal_list_valid)
deal_list_valid
```

The output that is list of deal ids can be passed to AdServer. 
