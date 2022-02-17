library(pROC)
library(glmnet)
library(randomForest)
library(mlbench)
library(tree)
data(Sonar)

N = nrow(Sonar)
P = ncol(Sonar) - 1
M = 150
set.seed(1)

mdata = Sonar[sample(1:N),]
itrain = sample(1:N,M)
#removes the col class
x = mdata[,-ncol(mdata)]
y = mdata$Class
xm = as.matrix(x)

#test = total-train
N-M

#fit lasso
lasso.cv = cv.glmnet(xm[itrain,], y[itrain], alpha=1,
                     family='binomial')
lasso.cv$lambda.min
lasso = glmnet(xm[itrain,], y[itrain], alpha=1,
               family='binomial', lambda=0.01)
head(coef(lasso),13)


#forests
tree.mod = tree(y~., data=x, subset=itrain)

#no of variables used in tree
length(summary(tree.mod)$used)
#names(summary(tree.mod))

#random forests
rf.mod = randomForest(y~., data=x, subset=itrain)
varImpPlot(rf.mod)


#predictions from classification tree
tree.pred = predict(tree.mod, x[-itrain,], 'class')
rf.pred = predict(rf.mod, x[-itrain,], 'class')

#confusion matrices
(tb.tree = table(tree.pred, y[-itrain]))
(tb.rf = table(rf.pred, y[-itrain]))

#error rates
1-sum(diag(tb.tree))/ sum(tb.tree)
1-sum(diag(tb.rf))/ sum(tb.rf)

#AUC
tree.p = predict(tree.mod, x[-itrain,], 'vector')[,2]
rf.p = predict(rf.mod, x[-itrain,], 'prob')[,2]
length(tree.p)
auc.tree = roc(y[-itrain], tree.p)$auc
auc.rf = roc(y[-itrain], rf.p)$auc
c(auc.tree, auc.rf)
