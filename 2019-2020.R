library(ISLR)
library(randomForest)
library(pROC)
library(gbm)

set.seed(4061)
n = nrow(Caravan)
dat = Caravan[sample(1:n, n, replace=F),]
dat$Purchase = as.factor(as.numeric(dat$Purchase=='Yes'))

i.train = sample(1:n, round(.7*n), replace=F)
x.train = dat[i.train, -ncol(dat)]
y.train = dat$Purchase[i.train]
x.test = dat[-i.train, -ncol(dat)]
y.test = dat$Purchase[-i.train]

######
#glm
glm.fit =  glm(Purchase~., data=dat, subset=i.train, family='binomial')
glm.pred = predict(glm.fit, newdata=dat[-i.train,], type='response')

#summary
summary(glm.fit)
summary(glm.pred)
tb = table(pred, y.test)
err = NULL
for(cut.off in seq(.1, .9, by=.1)){
  pred.y = as.numeric(glm.pred > cut.off)
  tb = table(pred.y, y.test)
  err = c(err, (1-sum(diag(tb))/sum(tb)))
}

#confusion matrix
tb
plot(seq(.1, .9, by=.1), err, t='b')

#error rate
1-sum(diag(tb))/ sum(tb)

######
#random forest
#random forest of 100 trees
rf.tree = randomForest(Purchase~., data=dat, subset=i.train, ntree=100)
rf.pred = predict(rf.tree, dat[-i.train,], type='class')

#confusion matrix
tb.rf = table(rf.pred, dat$Purchase[-i.train])
tb.rf

#error rate
1-sum(diag(tb.rf))/ sum(tb.rf)


######
#gbm
#