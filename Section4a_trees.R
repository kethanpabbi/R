# --------------------------------------------------------
# ST4061 / ST6041
# 2021-2022
# Eric Wolsztynski
# ...
# Exercises Section 4: Tree-based methods
# --------------------------------------------------------

###############################################################
### Exercise 1: growing and pruning a tree
###############################################################

library(ISLR) # contains the dataset
library(tree) # contains... tree-building methods

attach(Carseats)
# Recode response variable so as to make it a classification problem
High = ifelse(Carseats$Sales<=8, "0", "1")
# Create a data frame that includes both the predictors and response
# (a data frame is like an Excel spreadsheet, if you like)
CS = data.frame(Carseats, High)

# Fit the tree using all predictors (except for variable Sales, 
# which we have "recoded" into a cateorical response variable)
# to response variable High
tree.out = tree(High~.-Sales, CS)
cv.tree(tree.out, prune.tree)
summary(tree.out)
# plot the tree
plot(tree.out)
text(tree.out, pretty=1)

# pruning:
set.seed(3)
cv.CS = cv.tree(tree.out, FUN=prune.misclass)
names(cv.CS)
# - size:
# number of terminal nodes in each tree in the cost-complexity pruning sequence.
# - deviance:	
# total deviance of each tree in the cost-complexity pruning sequence.
# - k:
# the value of the cost-complexity pruning parameter of each tree in the sequence.
cv.CS
par(mfrow=c(1,2))
plot(cv.CS$size,cv.CS$dev,t='b')
plot(cv.CS$k,cv.CS$dev,t='b')

# use pruning: 
# - use which.min(cv.CS$dev) to get the location of the optimum
# - retrieve the corresponding tree size
# - pass this information on to pruning function
opt.size = cv.CS$size[which.min(cv.CS$dev)]
ptree = prune.misclass(tree.out, best=opt.size)
ptree 
summary(ptree)
par(mfrow=c(1,2))
plot(tree.out)
text(tree.out, pretty=0)
plot(ptree)
text(ptree, pretty=0)


############################################################### 
### Exercise 2: apply CV and ROC analysis
############################################################### 

# Train/test:
set.seed(2)
n = nrow(CS)
itrain = sample(1:n, 200)
CS.test = CS[-itrain,]
High.test = High[-itrain]
# argument 'subset' makes it easy to handle training/test splits:
tree.out = tree(High~.-Sales, CS, subset=itrain)
summary(tree.out)
plot(tree.out)
text(tree.out, pretty=0)

# prediction from full tree:
tree.pred = predict(tree.out, CS.test, type="class")
(tb1 = table(tree.pred,High.test))

# prediction from pruned tree:
ptree.pred = predict(ptree, CS.test, type="class")
(tb2 = table(ptree.pred,High.test)) # confusion matrix
sum(diag(tb1))/sum(tb1)
sum(diag(tb2))/sum(tb2)

# perform ROC analysis
library(pROC)
# here we specify 'type="vector"' to retrieve continuous scores
# as opposed to predicted labels, so that we can apply varying
# threshold values to these scores to build the ROC curve:
ptree.probs = predict(ptree, CS.test, type="vector")
roc.p = roc(response=(High.test), predictor=ptree.probs[,1])
roc.p$auc
plot(roc.p)
