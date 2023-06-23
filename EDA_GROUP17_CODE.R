install.packages("dplyr")
install.packages("caret")
install.packages("e1071")
install.packages("ggplot2")
install.packages("caTools")
install.packages("ROSE")
install.packages("smotefamily")
install.packages("rpart")
install.packages("rpart.plot")
install.packages("DMwR")
install.packages("adabag")
install.packages("e1071")
install.packages("caTools")
install.packages("caret")


# Importing required libraries
library(dplyr)
library(caret)
library(ggplot2)
library(caTools)
library(ROSE)
library(smotefamily)
library(rpart)
library(rpart.plot)
library(corrplot)
library(e1071)
library(randomForest)
library(adabag)

#Loading the dataset
credit_card=read.csv("E:/CSE3506 J-COMP/creditcard.csv")

#Viewing dataset
View(credit_card)

# Glance at the structure of the dataset
str(credit_card)

# Number of rows & columns
nrow(credit_card)
ncol(credit_card)


# Convert class to a factor variable
credit_card$Class = as.factor(credit_card$Class)

# Get the summary of the data
summary(credit_card)

# Count the missing values
sum(is.null(credit_card))

# Get the distribution of fraud and legit transactions in the dataset
class_count=table(credit_card$Class)


# Get the percentage of fraud and legit transactions in the datasets
prop.table(table(credit_card$Class))


# Histograms for each variable

par(mfrow = c(3,5))
i = 1
for (i in 1:30) {
  hist((credit_card[,i]), main = paste("Distibution of ", 
                                       colnames(credit_card[i])), 
       xlab = colnames(credit_card[i]),
       col = "light blue")
}

# correlation plot
par(mfrow=c(1,1))

r = cor(credit_card[,1:30])
corrplot(r, tl.col = 'black')


pie(class_count,main="credit card fraud")

ggplot(data = credit_card , aes(x = V1, y = V2, col =Class))+
  geom_point(position = position_jitter(width = 0.2))+
  theme_bw()+
  scale_color_manual(values = c('dodgerblue2','red'))


#ADASYN Balanced

train_adas = ADAS(credit_card[,-31],credit_card$Class,K = 5)
train_adas = train_adas$data  # extract only the balanced dataset
train_adas$class = as.factor(train_adas$class)

prop.table(table(train_adas$class))
summary(train_adas)
class_count=table(train_adas$class)
pie(class_count,main="credit card fraud")

# SMOTE Balanced

train_smote = SMOTE(credit_card[,-31],credit_card$Class,K = 5)
train_smote = train_smote$data # extract only the balanced dataset
train_smote$class = as.factor(train_smote$class)

prop.table(table(train_smote$class))

# train and test split for ADASYN

set.seed(1337)
train_split_adas = createDataPartition(train_adas$class,p = 0.7,times = 1,list = F)
train_data_adas = train_adas[ train_split_adas,]
test_data_adas  = train_adas[-train_split_adas,]

prop.table(table(train_data_adas$class))

prop.table(table(test_data_adas$class))


# Train test split for SMOTE

set.seed(1337)
train_split_smote = createDataPartition(train_smote$class,p = 0.7,times = 1,list = F)
train_data_smote = train_smote[ train_split_smote,]
test_data_smote  = train_smote[-train_split_smote,]

prop.table(table(train_data_smote$class))

prop.table(table(test_data_smote$class))


# Decision Tree 

# using ADASYN

dt_adas = rpart(class ~ ., data = train_data_adas,method = "class")
rpart.plot(dt_adas)
dt_pred_adas =predict(dt_adas, test_data_adas, type = 'class')
dt_cm_adas = table(test_data_adas$class, dt_pred_adas)
confusionMatrix(dt_cm_adas)

# using SMOTE

dt_smote = rpart(class ~ ., data = train_data_smote, method = "class")
rpart.plot(dt_smote)
dt_pred_smote =predict(dt_smote, test_data_smote, type = 'class')
dt_cm_smote = table(test_data_smote$class, dt_pred_smote)
confusionMatrix(dt_cm_smote)

# Navie Bayes

# using ADASYN

naive_adas = naiveBayes(class ~ ., data = train_data_adas)
nb_pred_adas = predict(naive_adas, newdata = test_data_adas)
nb_cm_adas = table(test_data_adas$class, nb_pred_adas)
confusionMatrix(nb_cm_adas)

# using SMOTE

naive_smote = naiveBayes(class ~ ., data = train_data_smote)
nb_pred_smote = predict(naive_smote, newdata = test_data_smote)
nb_cm_smote = table(test_data_smote$class, nb_pred_smote)
confusionMatrix(nb_cm_smote)


# LDM

library(MASS)

#using ADASYN

lda_adas=lda(class ~ .,data=train_data_adas)
lda_pred_adas=predict(lda_adas,newdata=test_data_adas)
lda_cm_adas=table(test_data_adas$class,lda_pred_adas$class)
confusionMatrix(lda_cm_adas)


lda_adas_plot = cbind(train_data_adas, predict(lda_adas)$x)
ggplot(lda_adas_plot, aes(V1, V2)) +
  geom_point(aes(color = class))

# using SMOTE


lda_smote=lda(class ~ .,data=train_data_smote)
lda_pred_smote=predict(lda_smote,newdata=test_data_smote)
lda_cm_smote=table(test_data_smote$class,lda_pred_smote$class)
confusionMatrix(lda_cm_smote)



lda_smote_plot = cbind(train_data_smote, predict(lda_smote)$x)
ggplot(lda_smote_plot, aes(V1, V2)) +
  geom_point(aes(color = class))


# AdaBoost


library(adabag)
model_adas = boosting(class ~ ., data =train_data_adas, boos = TRUE, mfinal = 10)
pred = predict(model_adas, test_data_adas)
adaboost_cm_adas=table(test_data_adas$class,pred$class)
confusionMatrix(adaboost_cm_adas)

nrow(test_data_adas)

# Voting

final=list()
for (i in 1:170583){
  no=0
  yes=0
  if(pred$class[i]==0){
    no=no+1
  }
  else{
    yes=yes+1
  }
  if(lda_pred_adas$class[i]==0){
    no=no+1
  }
  else{
    yes=yes+1
  }
  if(dt_pred_adas[i]==0){
    no=no+1
  }
  else{
    yes=yes+1
  }
  if (yes>no){
    final=append(final,'1')
  }
  else{
    final=append(final,'0')
  }
}

final_pred=data.frame(class = unlist(final))

final_cm=table(test_data_adas$class,final_pred$class)
confusionMatrix(final_cm)


