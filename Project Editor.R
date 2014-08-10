# STATS202 project - Carl Johan Rising - August 2014

#Data attributes: 
# (1) query_id (2) url_id (3) query_length (4) is_homepage
# (5-12) sig1-8 (13) relevance [1/0]
# Training : 65535 rows Header=T - query_id 4631- data[,1:13]
# Test : ... rows Header=T data[,1:12] 

#On start
FullTrain <- read.csv("training.csv", header=TRUE)
class(FullTrain) #Should return: [1] "data.frame" 
indexes = sample(1:nrow(FullTrain), size=0.2*nrow(FullTrain))
maintest = FullTrain[indexes,]
maintrain = FullTrain[-indexes,]
trans_Train<-maintrain
trans_Train[,7:10][trans_Train[,7:10]==0] <- 0.001
trans_Train[,7:10] <- log(trans_Train[,7:10]) #transforms column 7:10 to log value
trans_Test <- maintest
trans_Test[,7:10][trans_Test[,7:10]==0] <- 0.001
trans_Test[,7:10] <- log(trans_Test[,7:10]) #transforms column 7:10 to log value
library("rpart")
library("class")
library("e1071")
install.packages("randomForest")
library("randomForest")

summary(trans_Test)

summary(trans_Train[,7:10])
summary(trans_Test[,7:10])
cor(trans_Test[,5:12])

# Load of data-sets
FullTrain <- read.csv("training.csv", header=TRUE)
FullTest <- read.csv("test.csv", header=TRUE)
testtest <- read.csv("untitled.csv",header=FALSE)

is.na(FullTrain) # Tests for missing values
is.na(FullTest) # Tests for missing values
is.na(testtest)

#Libraries
library("")
library("rpart")
library("class")
library("e1071")
install.packages("randomForest")
library("randomForest")
install.packages("corrgram")
library("corrgram")

# Samples

# Correlations
corrdata <- Sample1
corrgram(maintrain, main="Correlations for Training Data")
corrgram(maintest, main="Correlations for Test Data")
cor(FullTrain[,5:12])

cor(maintest[,5:12])
cor(maintrain[,5:12])
cor(trans_Train[,5:12])
is.factor(trans_Train[,10])

# Modify data
	# Split data
class(FullTrain) #Should return: [1] "data.frame" 
indexes = sample(1:nrow(FullTrain), size=0.2*nrow(FullTrain))
maintest = FullTrain[indexes,]
maintrain = FullTrain[-indexes,]

	#Count rows
nrow(FullTrain) #80046 - 100%
nrow(maintrain) #64037 - 80%
nrow(maintest) #16009 - 20%
nrow(trans_Test)
nrow(trans_Train)

	# Transform
trans_Train[,7:10][trans_Train[,7:10]==0] <- 0.001
trans_Train[,7:10] <- log(trans_Train[,7:10]) #transforms column 7:10 to log value
trans_Test[,7:10][trans_Test[,7:10]==0] <- 0.001
trans_Test[,7:10] <- log(maintest[,7:10]) #transforms column 7:10 to log value

# Plot data - sig1-8
hist(maintrain[,5],main="Histogram: sig1",xlab="Value",col="lightgreen") #sig1
hist(maintrain[,6],main="Histogram: sig2",xlab="Value",col="lightgreen") #sig2
hist(maintrain[,7],main="Histogram: sig3",xlab="Value",col="magenta") #sig3
hist(log(maintrain[,7]),main="Histogram: log(sig3)",xlab="Value",col="lightblue") #log(sig3)
hist(maintrain[,8],main="Histogram: sig4",xlab="Value",col="magenta") #sig4
hist(log(maintrain[,8]),main="Histogram: log(sig4)",xlab="Value",col="lightblue") #log(sig4)
hist(maintrain[,9],main="Histogram: sig5",xlab="Value",col="magenta") #sig5
hist(log(maintrain[,9]),main="Histogram: log(sig5)",xlab="Value",col="lightblue") #log(sig5)
hist(maintrain[,10],main="Histogram: sig6",xlab="Value",col="magenta") #sig6
hist(log(maintrain[,10]),main="Histogram: log(sig6)",xlab="Value",col="lightblue") #log(sig6)
hist(maintrain[,11],main="Histogram: sig7",xlab="Value",col="lightgreen") #sig7
hist(maintrain[,12],main="Histogram: sig8",xlab="Value",col="lightgreen") #sig8
hist(trans_Train[,])
##############################################
##############################################
# CLASSIFIERS
# Decision tree - rpart
# depth=5 test_error: 0.36123

y<-as.factor(trans_Train[,13])
x<-trans_Train[,3:12]
y_test<-as.factor(trans_Test[,13])
x_test<-trans_Test[,3:12]
i<-1
dep<-i
for (i in 1:10) {
	fit<-rpart(y~.,x,control=rpart.control(minsplit=0,minbucket=0,cp=-1,maxcompete=0,maxsurrogate=0, usesurrogate=0,xval=0,maxdepth=i))
	train_error[i]<-sum(y!=predict(fit,x,type="class"))/length(y)
	test_error[i]<-sum(y_test!=predict(fit,x_test,type="class"))/length(y_test)
	dep[i]<-i
	i<-i+1
}

plot(dep,train_error,type="l",pch=19, ylim=c(0,1), col="blue", xlab="Depth of tree", ylab="Error rate", main="Carl Johan Rising")
par(new=T)
plot(dep,test_error,type="l",pch=19, ylim=c(0,1), col="black", xlab="Depth of tree", ylab="Error rate", main="Carl Johan Rising")
legend(7,.8,c("Train error","Test error"),col=c("blue","black"),pch=19)	
min(test_error)
test_error

# SVM - Support Vector Machine 
# Result: trans_Test: .3354/.3309 [,3:12]
# for maintrain and maintest: test errorrate: .365, cost=1
#train<-read.csv("training.csv",header=TRUE) 
# second test: .364 [1:1000,3:12] -> next .3309 / .3354
i<-1
cost<-i
for(i in 1:10){
y<-as.factor(trans_Train[,13])
x<-trans_Train[,3:12]
y_test<-as.factor(trans_Test[,13])
x_test<-trans_Test[,3:12]
fit<-svm(x,y,cost=1)
train_error[i]<-1-sum(y==predict(fit,x))/length(y)
test_error[i]<-1-sum(y_test==predict(fit,x_test))/length(y_test)
cost[i]<-i
i<-i+1
}
cost
i
plot(cost,test_error, type="l")
train_error

# KNN - K Nearest Neighbour 
# Results: trans_Test: 0.4346305 for k=5
# for maintest: .448
	y<-as.factor(maintrain[1:1000,13])
	x<-maintrain[1:1000,3:12]
	y_test<-as.factor(maintest[1:1000,13])
	x_test<-maintest[1:1000,3:12]
	i<-1
	fit_test<-knn(x,x_test,y,k=1)
	fit<-knn(x,x,y,k=5)
	train_error<-
	1-sum(y==fit)/length(y)
	test_error<-
	1-sum(y_test==fit_test)/length(y_test)
	i<-i+1
summary(x,x_test,y)


for (i in 1:10){
	y<-as.factor(trans_Train[,13])
	x<-trans_Train[,3:12]
	y_test<-as.factor(trans_Test[,13])
	x_test<-trans_Test[,3:12]
	fit_test<-knn(x,x_test,y,k=i)
	fit<-knn(x,x,y,k=i)
	train_error[i] <- 1-sum(y==fit)/length(y)
	test_error[i] <- 1-sum(y_test==fit_test)/length(y_test)
	i<-i+1
} 
print(test_error)
plot(train_error,type="l",pch=19, col="blue",main="Carl Johan Rising",ylim=c(0,1),ylab="Error rate",xlab="k value") 
par(new=T)
plot(test_error,type="l",pch=19, col="black",main="Carl Johan Rising",ylim=c(0,1),ylab="Error rate",xlab="k value")
legend(2,.8,c("Train error","Test error"),col=c("blue","black"),lty=1,pch=19)

	
# NaiveBayes
?naiveBayes()

naive_model   <- naiveBayes(trans_Train[1:1000, -c(13)], trans_Train[1:1000, 13], laplace = 1)
naive_predict <- predict(naive_model, trans_Train[60001:80000, -c(13)], type=c("class"))
summary(naive_predict)

1-sum(y_test==predict(naive_model,x_test))/length(y_test)

pairs(trans_Train[1:10,5:12],col=c("red","blue","lightgreen","lightblue","green","cyan","magenta","purple"))
naiveBayes(trans_Train,)

classifier<-naiveBayes(trans_Train[,5:12], trans_Train[,13]) 
table(predict(classifier, trans_Train[,-13]), trans_Train[,13])
classifier 


# Ensemble methods:
# RandomForest testerror: .38 [1:1000,3:12]

y<-as.factor(trans_Train[1:1000,13])
x<-trans_Train[1:1000,3:12]
y_test<-as.factor(trans_Test[1:1000,13])
x_test<-trans_Test[1:1000,3:12]
fit<-randomForest(x,y) 
1-sum(y_test==predict(fit,x_test))/length(y_test)


# AdaBoost
?rep
y<-trans_Train[,13]
x<-trans_Train[,3:12]
y_test<-trans_Test[,13]
x_test<-trans_Test[,3:12]
train_error<-rep(0,500) # Keep track of errors 
test_error<-rep(0,500)
f<-rep(0,130) # 130 pts in training data
f_test<-rep(0,78) # 78 pts in test data
i<-1
library(rpart)
while(i<=500){
	w<-exp(-y*f) # This is a shortcut to compute w + w<-w/sum(w)
	fit<-rpart(y~.,x,w,method="class")
	g<-1+2*(predict(fit,x)[,2]>.5) # make -1 or 1 
	g_test<-1+2*(predict(fit,x_test)[,2]>.5)
	e<-sum(w*(y*g<0))
	alpha<-.1*log ( (1-e) / e ) # or .5/.1
	f<-f+alpha*g
	f_test<-f_test+alpha*g_test
	train_error[i]<-sum(1*f*y<0)/130
	test_error[i]<-sum(1*f_test*y_test<0)/78
	i<-i+1
}

plot(seq(1,500),test_error,type="l",ylim=c(0,.5),ylab="Error Rate",xlab="Iterations",lwd=2,main="Carl Johan Rising")
lines(train_error,lwd=2,col="purple")
legend(4,.5,c("Training Error","Test Error"), col=c("purple","black"),lwd=2)


#########
train_error<-rep(0,500) # Keep track of errors 
test_error<-rep(0,500)
w_train<-rep(0,500)
w_test<-rep(0,500)

while(i<=500){
	w_train = exp(-y_sonar_train*f_train)
	train_exp_loss[i] = log(sum(w_train))
	w_test = exp(-y_sonar_test*f_test)
	test_exp_loss[i] = log(sum(w_test))
	w = w_train/sum(w_trai n)
	fit_Boost = rpart(y_sonar_train~.,x_sonar_train,w,method="class")
	g = -1+2*(predict(fit_Boost,x_sonar_train)[,2]>.5)
	g_test = -1+2*(predict(fit_Boost,x_sonar_test)[,2]>.5)
	e = sum(w*(y_sonar_train*g < 0))
	alpha = .5*log ( (1-e) / e )
	f_train = f_train + alpha * g
	f_test = f_test + alpha * g_test
	train_error_Boost[i] = sum(1*f_train*y_sonar_train<0)/130
	test_error_Boost[i] = sum(1*f_test*y_sonar_test<0)/78
	i = i+1
}


