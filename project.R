setwd("C:\\Users\\Saurav Singla\\Documents\\Reema\\Pactical Machine Learning")
training_data = read.csv("pml-training.csv")
testing_data = read.csv("pml-testing.csv")

#Remove columns with more than 20% missing values
maxNAallowed = ceiling(nrow(training_data)/100 * 20)
removeColumns = which(colSums(is.na(training_data)| training_data=="")>maxNAallowed)
training_data_clean = training_data[,-removeColumns]
testing_data_clean = testing_data[,-removeColumns]

#remove time related columns
remove_time = grep("timestamp",names(training_data_clean))
training_without_time = training_data_clean[,-c(1,remove_time)]
testing_without_time = testing_data_clean[,-c(1,remove_time)]

#convert target factor variable("classe") into integer
training_clas_int <- data.frame(data.matrix(training_without_time))
testing_clas_int <- data.frame(data.matrix(testing_without_time))

#final data
train_data = training_clas_int
testing_data = testing_clas_int

#split train data into test and train
set.seed(18765277)
library(caret)
classeIndex <- which(names(train_data) == "classe")
partition <- createDataPartition(y=train_data$classe, p=0.75, list=FALSE)
train_sub_Train <- train_data[partition, ]
train_sub_Test <- train_data[-partition, ]

#correlation with target variable
correlations <- cor(train_sub_Train[, -classeIndex], train_sub_Train$classe)
bestCorrelations <- subset(as.data.frame(as.table(correlations)), abs(Freq)>0.29)
bestCorrelations

#plot the correlation with target variable
library(Rmisc)
p1 <- ggplot(train_sub_Train, aes(x=classe,y=pitch_forearm, fill=factor(classe))) + 
  geom_boxplot(stat = 'boxplot', aes(group=classe))
p2 <- ggplot(train_sub_Train, aes(classe, magnet_arm_x, fill=factor(classe))) + 
  geom_boxplot(stat = 'boxplot', aes(group=classe))
p3 <- ggplot(train_sub_Train, aes(classe, magnet_belt_y, fill=factor(classe))) + 
  geom_boxplot(stat = 'boxplot', aes(group=classe))
multiplot(p1,p2,p3)

#bi-variate analysis
correlationMatrix <- cor(train_sub_Train[, -classeIndex])
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.9, exact=TRUE)
excludeColumns <- c(highlyCorrelated, classeIndex)
library(corrplot)
corrplot(correlationMatrix, method="color", type="lower", order="hclust", tl.cex=0.70, tl.col="black", tl.srt = 45, diag = FALSE)
library(factoextra)
library(FactoMineR)

#removing highly correlated variables using PCA (Principle Component Analysis)
pcaPreProcess <- preProcess(train_sub_Train[, -classeIndex], method = "pca", thresh = 0.99)
train_sub_Train_pca <- predict(pcaPreProcess, train_sub_Train[, -classeIndex])
train_sub_Test_pca <- predict(pcaPreProcess, train_sub_Test[, -classeIndex])
test_pca <- predict(pcaPreProcess, testing_data[, -classeIndex])

pcaPreProcess_subset <- preProcess(train_sub_Train[, -excludeColumns], method = "pca", thresh = 0.99)
train_sub_Train_pca_sub <- predict(pcaPreProcess_subset, train_sub_Train[, -excludeColumns])
train_sub_Test_pca_sub <- predict(pcaPreProcess_subset, train_sub_Test[, -excludeColumns])
test_pca_sub <- predict(pcaPreProcess_subset, testing_data[, -classeIndex])

#Random_Forest Model 1
library(randomForest)
ntree <- 200 
start <- proc.time()
rfMod.cleaned <- randomForest(
  x=train_sub_Train[, -classeIndex], 
  y=train_sub_Train$classe,
  xtest=train_sub_Test[, -classeIndex], 
  ytest=train_sub_Test$classe, 
  ntree=ntree,
  keep.forest=TRUE,
  proximity=TRUE)
proc.time() - start

#Random_Forest Model 2
start <- proc.time()
rfMod.exclude <- randomForest(
  x=train_sub_Train[, -excludeColumns], 
  y=train_sub_Train$classe,
  xtest=train_sub_Test[, -excludeColumns], 
  ytest=train_sub_Test$classe, 
  ntree=ntree,
  keep.forest=TRUE,
  proximity=TRUE)
proc.time() - start

#Random_Forest Model 3
start <- proc.time()
rfMod.pca.all <- randomForest(
  x=train_sub_Train_pca, 
  y=train_sub_Train$classe,
  xtest=train_sub_Test_pca, 
  ytest=train_sub_Test$classe, 
  ntree=ntree,
  keep.forest=TRUE,
  proximity=TRUE) 
proc.time() - start

#Random_Forest Model 4
start <- proc.time()
rfMod.pca.subset <- randomForest(
  x=train_sub_Train_pca_sub, 
  y=train_sub_Train$classe,
  xtest=train_sub_Test_pca_sub, 
  ytest=train_sub_Test$classe, 
  ntree=ntree,
  keep.forest=TRUE,
  proximity=TRUE) 
proc.time() - start

#Accuracy of Model 1
rfMod.cleaned
rfMod.cleaned.training.acc <- round(1-sum(rfMod.cleaned$confusion[, 'class.error']),3)
paste0("Accuracy on training: ",rfMod.cleaned.training.acc)
rfMod.cleaned.testing.acc <- round(1-sum(rfMod.cleaned$test$confusion[, 'class.error']),3)
paste0("Accuracy on testing: ",rfMod.cleaned.testing.acc)

#Accuracy of Model 2
rfMod.exclude
rfMod.exclude.training.acc <- round(1-sum(rfMod.exclude$confusion[, 'class.error']),3)
paste0("Accuracy on training: ",rfMod.exclude.training.acc)
rfMod.exclude.testing.acc <- round(1-sum(rfMod.exclude$test$confusion[, 'class.error']),3)
paste0("Accuracy on testing: ",rfMod.exclude.testing.acc)

#Accuracy of Model 3
rfMod.pca.all
rfMod.pca.all.training.acc <- round(1-sum(rfMod.pca.all$confusion[, 'class.error']),3)
paste0("Accuracy on training: ",rfMod.pca.all.training.acc)
rfMod.pca.all.testing.acc <- round(1-sum(rfMod.pca.all$test$confusion[, 'class.error']),3)
paste0("Accuracy on testing: ",rfMod.pca.all.testing.acc)

#Accuracy of Model 4
rfMod.pca.subset
rfMod.pca.subset.training.acc <- round(1-sum(rfMod.pca.subset$confusion[, 'class.error']),3)
paste0("Accuracy on training: ",rfMod.pca.subset.training.acc)
rfMod.pca.subset.testing.acc <- round(1-sum(rfMod.pca.subset$test$confusion[, 'class.error']),3)
paste0("Accuracy on testing: ",rfMod.pca.subset.testing.acc)

#This concludes that nor PCA doesn't have a positive of the accuracy (or the process time for that matter)
#The "rfMod.exclude" perform's slightly better then the "rfMod.cleaned"
#We'll stick with the `rfMod.exclude` model as the best model to use for predicting the test set.
#Because with an accuracy of 98.7% and an estimated OOB error rate of 0.23% this is the best model.

#Before doing the final prediction we will examine the chosen model more in depth using some plots
par(mfrow=c(1,2)) 
varImpPlot(rfMod.exclude, cex=0.7, pch=16, main='Variable Importance Plot: rfMod.exclude')
plot(rfMod.exclude, , cex=0.7, main='Error vs No. of trees plot')
par(mfrow=c(1,1)) 

#To look at the distances between predictions we can use MDSplot and cluster predictions and results
start <- proc.time()
library(RColorBrewer)
palette <- brewer.pal(length(classeLevels), "Set1")
rfMod.mds <- MDSplot(rfMod.exclude, as.factor(classeLevels), k=2, pch=20, palette=palette)
library(cluster)
rfMod.pam <- pam(1 - rfMod.exclude$proximity, k=length(classeLevels), diss=TRUE)
plot(
  rfMod.mds$points[, 1], 
  rfMod.mds$points[, 2], 
  pch=rfMod.pam$clustering+14, 
  col=alpha(palette[as.numeric(train_sub_Train$classe)],0.5), 
  bg=alpha(palette[as.numeric(train_sub_Train$classe)],0.2), 
  cex=0.5,
  xlab="x", ylab="y")
legend("bottomleft", legend=unique(rfMod.pam$clustering), pch=seq(15,14+length(classeLevels)), title = "PAM cluster")
legend("topleft", legend=classeLevels, pch = 16, col=palette, title = "Classification")
proc.time() - start

#Let's look at predictions for all models on the final test set. 

predictions <- t(cbind(
  exclude=as.data.frame(predict(rfMod.exclude, testing_data[, -excludeColumns]), optional=TRUE),
  cleaned=as.data.frame(predict(rfMod.cleaned, testing_data), optional=TRUE),
  pcaAll=as.data.frame(predict(rfMod.pca.all, testing.pca.all), optional=TRUE),
  pcaExclude=as.data.frame(predict(rfMod.pca.subset, testing.pca.subset), optional=TRUE)
))
predictions

#The predictions don't really change a lot with each model,
#but since we have most faith in the "rfMod.exclude", we'll keep that as final answer. 


