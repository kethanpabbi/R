# --------------------------------------------------------
# ST4061 / ST6041
# 2021-2022
# Eric Wolsztynski
# ...
# Exercises Section 1: Daat pre-processing
# --------------------------------------------------------

library(glmnet)
library(survival)
library(ISLR)

###############################################################
### Exercise 1: effect of scaling
###############################################################

dat = iris[1:100,]
dat$Species = droplevels(dat$Species)
x = dat[,1:4]
y = dat$Species
# we can also apply scaling to the x data directly:
dats = dat
dats[,1:4] = apply(dats[,1:4],2,scale)
xs = apply(x,2,scale)

# (1)
pca.unscaled = prcomp(x)
pca.scaled = prcomp(x,scale=TRUE)
pca.scaled.2 = prcomp(xs) # should be the same as pca.scaled

# plot the data on its first 2 dimensions in each space: 
par(mfrow=c(1,3))
plot(x[,1:2], pch=20, col=y, main="Data in original space") 
biplot(pca.unscaled, main="Data in PCA space")
abline(v=0, col='orange')
# see the binary separation in orange along PC1? 
# re-use this into biplot in original space:
pca.cols = c("blue","orange")[1+as.numeric(pca.unscaled$x[,1]>0)]
plot(x[,1:2], pch=20, col=pca.cols, 
	main="Data in original space\n colour-coded using PC1-split") 


par(mfrow=c(2,2))
plot(pca.unscaled) # scree plot 
biplot(pca.unscaled) # biplot 
plot(pca.scaled) # scree plot 
biplot(pca.scaled) # biplot 
# now analyse this plot :)

# (2)
logreg.unscaled = glm(Species~., data=dat) # make this work
logreg.unscaled = glm(Species~., data=dat, family='binomial')
logreg.scaled = glm(Species~., data=dats, family='binomial')
# discuss... 
# (... are the fits different?)
cbind(coef(logreg.unscaled), coef(logreg.scaled))
# (... does this align with the PCA analysis?)
# (yes, we see a change in the role of each variable in the
#  information space)
# (... and why?)
# (both linear techniques)

# (3)
x.m = model.matrix(Species~.+0, data=dat)
lasso.cv = cv.glmnet(x.m, y, family="binomial")
lasso.unscaled = glmnet(x.m, y, family="binomial", lambda=lasso.cv$lambda.min)
lasso.pred = predict(lasso.unscaled, newx=x.m, type="class")
#
xs.m = model.matrix(Species~.+0, data=dats)
lasso.cv = cv.glmnet(xs.m, y, family="binomial")
lasso.scaled = glmnet(xs.m, y, family="binomial", lambda=lasso.cv$lambda.min)
lasso.s.pred = predict(lasso.scaled, newx=xs.m, type="class")
#
cbind(coef(lasso.unscaled), coef(lasso.scaled))
table(lasso.pred, lasso.s.pred) # meh

###############################################################
### Exercise 2: data imputation
###############################################################

summary(lung)
boxplot(lung$meal.cal~lung$sex, col=c("cyan","pink"))
# can you think of other ways of analysing this?

# (1) lung cancer data: compare meal.cal values between male and female cohorts, 
# and discuss w.r.t. gender-specific data imputation 
# NB: "missing at random" vs "missing not at random"??

nas = is.na(lung$meal.cal) # track missing values
table(nas, lung$sex)
imales = which(lung$sex==1)
m.all = mean(lung$meal.cal, na.rm=TRUE)
m.males = mean(lung$meal.cal[imales], na.rm=TRUE)
m.females = mean(lung$meal.cal[-imales], na.rm=TRUE)
t.test(lung$meal.cal[imales], lung$meal.cal[-imales])
# significant difference, hence must use different imputation
# values for each gender

# (2) Run Cox PHMs on original, overall imputed and gender-specific imputed
# datsets, using the cohort sample mean for data imputation. Compare and discuss 
# model fitting output.

dat1 = dat2 = lung
# impute overall mean in dat1:
dat1$meal.cal[nas] = m.all 
# impute gender=sepcific mean in dat2:
dat2$meal.cal[(is.na(lung$meal.cal) & (lung$sex==1))] = m.males
dat2$meal.cal[(is.na(lung$meal.cal) & (lung$sex==2))] = m.females
cox0 = coxph(Surv(time,status)~.,data=lung) 
cox1 = coxph(Surv(time,status)~.,data=dat1) 
cox2 = coxph(Surv(time,status)~.,data=dat2)
summary(cox0)
summary(cox1)
summary(cox2)
# - dat1 and dat2 yield increased sample size (from 167 to 209, both imputed 
# datasets having 209 observations)
# - overall coefficient effects comparable between the 2 sets
# - marginal differences in covariate effect and significance between lung and {dat1;dat2}
# - no substantial difference between dat1 and dat2 outputs

###############################################################
### Exercise 3: data imputation
###############################################################

library(ISLR)
dat = Hitters

# (1) (Deletion)
sdat = na.omit(dat)
sx = model.matrix(Salary~.+0,data=sdat)
sy = sdat$Salary
cv.l = cv.glmnet(sx,sy)
slo = glmnet(sx,sy,lambda=cv.l$lambda.min)

# (2) Simple imputation (of Y) using overall mean
ina = which(is.na(dat$Salary))
dat$Salary[ina] = mean(dat$Salary[-ina])
x = model.matrix(Salary~.+0,data=dat)
y = dat$Salary
cv.l = cv.glmnet(x,y)
lo = glmnet(x,y,lambda=cv.l$lambda.min)

# (3)
slop = predict(slo,newx=sx)
lop = predict(lo,newx=x)
sqrt(mean((slop-sy)^2))
sqrt(mean((lop-y)^2))
plot(slop,lop[-ina])
abline(a=0,b=1)
abline(lm(lop[-ina]~slop), col='navy')

# What could we do instead of imputing the Y?

###############################################################
### Exercise 4: resampling
###############################################################

# You don't need help for this one!

###############################################################
### Exercise 5: resampling (CV vs bootstrapping)
###############################################################

# Implement this simple analysis and discuss - think about 
# (sub)sample sizes!

x = trees$Girth   # sorted in increasing order...
y = trees$Height
plot(x, y, pch=20)
summary(lm(y~x))
N = nrow(trees)

# (1) 10-fold CV on original dataset
set.seed(4060)
K = 10
cc = numeric(K)
folds = cut(1:N, K, labels=FALSE)
for(k in 1:K){
	i = which(folds==k)
	# train:
	lmo = lm(y[-i]~x[-i])
	cc[k] = summary(lmo)$coef[2,2]
	# (NB: no testing here, so not the conventional use of CV)
}
mean(cc)

# (2) 10-fold CV on randomized dataset
set.seed(1)
mix = sample(1:nrow(trees), replace=FALSE)
xr = trees$Girth[mix]
yr = trees$Height[mix]
set.seed(4060)
K = 10
ccr = numeric(K)
folds = cut(1:N, K, labels=FALSE)
for(k in 1:K){
	i = which(folds==k)
	lmo = lm(yr[-i]~xr[-i])
	ccr[k] = summary(lmo)$coef[2,2]
}
mean(ccr)

sd(ccr)
sd(cc)
boxplot(cc,ccr)
t.test(cc,ccr)
var.test(cc,ccr)

# (3) Bootstrapping (additional note)
set.seed(4060)
K = 100
cb = numeric(K)
for(i in 1:K){
	# bootstrapping
	ib = sample(1:N,N,replace=TRUE)
	lmb = lm(y[ib]~x[ib])
	cb[i] = summary(lmb)$coef[2,2]
}
mean(cb)

dev.new()
par(font=2, font.axis=2, font.lab=2)
boxplot(cbind(cc,ccr,cb), names=c("CV","CVr","Bootstrap"))
abline(h=1.0544)
t.test(cc,cb)
# Explain why these are different?

round(cc, 3)
