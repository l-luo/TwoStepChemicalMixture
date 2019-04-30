rm(list=ls())
library(MASS)
library(selectiveInference)

library("xlsx")
library(nortest)
require(glmnet)
library(rtf)
library(randomForestSRC)
library(selectiveInference)
library(rms)
library(caret)
library(rpart)
library(DT)
## set up environment;
source("Adaptive_LASSO_Script_exm.R")

examine_mse<-function(mod) {
  ra<-(summary(mod))$residuals 
  (msea<-sum(ra^2)/length(ra) ) 
}
rand.seed=28993936  #sample(1:987654321,1)
set.seed(rand.seed)

## set up simulation
#no. predictors;
m=20
# sample size;
nsize=500
# itereration;
niter=1000

# generate replicates (predictors) based on multinormal distribution
mu=rep(1,m)
Sigma=diag(rep(1,m))

rho1=0.1
Sigma[1:15,1:15]<-Sigma[1:15,1:15]+matrix(rho1,15,15)-diag(rep(rho1,15))
rho2=0.05
Sigma[16:20,16:20]<-Sigma[16:20,16:20]+matrix(rho2,5,5)-diag(rep(rho2,5))

## try one iteration
mat1<-mvrnorm(n=nsize,mu,Sigma)
colnames(mat1)<-paste("V",1:20,sep="")
beta1=0.2 
beta2=0.3 
beta12=0.2 
beta1_12=0.1 
beta9=-0.15 
beta15=0.10 
beta16=-0.25 


strbeta<-"
beta1=0.2 
beta2=0.3 
beta12=0.2 
beta1_12=0.1 
beta9=-0.15 
beta15=0.10 
beta16=-0.25 
"
Y1<-beta1*mat1[,1]+beta2*mat1[,2]+beta12*mat1[,12]+beta1_12*mat1[,1]*mat1[,12]+beta9*mat1[,9]+
  beta15*mat1[,15]+beta16*mat1[,16]+rnorm(nsize,0,1)

df<-as.data.frame(cbind(mat1,Y1))
summary(lm(Y1~.,data=df))

## df dataset ready for simulation
## 



#########################################################
## ONE STEP Adaptive Lasso NO interaction
#########################################################

fita<-model_adaptive_LASSO_sel(pred=mat1,outcome=Y1 )
#summary(fita)$coefficients
msea=examine_mse(fita)


Coefmat1<-signif(summary(fita)$coefficients,2) 
beta_se1=cbind(paste(Coefmat1[,1],"(",Coefmat1[,2] ,")",sep="") ,Coefmat1[,4])
rownames(beta_se1)=rownames(Coefmat1)

#########################################################
## Setup 10fold CV;
#########################################################

folds<-createFolds(df$Y1, k = 10, list = TRUE, returnTrain = TRUE)
mspea.cv<-rep(NA,10)
msea.CV<-rep(NA,10)
nsamps=dim(df)[1]
pred.outcome<-rep(NA,nsamps)

for (kfold in 1:10) {
  idk=folds[[kfold]]
  fitK<-model_adaptive_LASSO_sel(pred=mat1[idk,],outcome=Y1[idk])
  msea.CV[kfold]=examine_mse(fitK)
  pred.obj<-predict(fitK, newdata=as.data.frame(mat1[-idk,]), se.fit = TRUE)
  pred.outcome[-idk]=pred.obj$fit
  
  rm(idk,fitK,pred.obj)
}

mspea.CV<-sum((pred.outcome-Y1)^2)/length(Y1)
#mean(msea.CV),mspea.CV


#########################################################
## chunk2: CART +Adaptive Lasso
#########################################################
cart.fit<-rpart(Y1~.,data=df)
plot(cart.fit)
text(cart.fit, use.n = TRUE)
cimp<-sort(cart.fit$variable.importance, decreasing = TRUE)
# arbitarily keep top half variables;
sel.vars.bycart<- names(cimp)[1:(m/2)]
fitb<-model_adaptive_LASSO_sel(pred=mat1[,sel.vars.bycart],outcome=Y1 )
mseb=examine_mse(fitb)
Coefmat2<-signif(summary(fitb)$coefficients,2) 
beta_se2=cbind(paste(Coefmat2[,1],"(",Coefmat2[,2] ,")",sep="") ,Coefmat2[,4])
rownames(beta_se2)=rownames(Coefmat2)

### performance cal. based on cross.validation;

mseb.CV<-rep(NA,10)
nsamps=dim(df)[1]
pred.outcome<-rep(NA,nsamps)

for (kfold in 1:10) {
  idk=folds[[kfold]]
  fitK<-model_adaptive_LASSO_sel(pred=mat1[idk,sel.vars.bycart],outcome=Y1[idk])
  mseb.CV[kfold]=examine_mse(fitK)
  pred.obj<-predict(fitK, newdata=as.data.frame(mat1[-idk,sel.vars.bycart]), se.fit = TRUE)
  pred.outcome[-idk]=pred.obj$fit
  
  rm(idk,fitK,pred.obj)
}
mspeb.CV<-sum((pred.outcome-Y1)^2)/length(Y1)


#########################################################
## chunk3: Neural Network
#########################################################

#########################################################
## Chumk 4: Two step- RF +adaptive lasso, no interaction
#########################################################

## Step1- RF selection;
set.seed(12345678)
n.trees=5000
rf.obj <- rfsrc(Y1 ~ ., data = df,ntree =n.trees,tree.err=TRUE,importance=TRUE)
Vars.subset<-intersect(names(which(rf.obj$importance>0)),
                       names(sort(rf.obj$importance,decreasing = TRUE))[1:(m/2)] )

## Adaptive Lasso 

fitc<-model_adaptive_LASSO_sel(pred=mat1[,Vars.subset],outcome=Y1)
(Coefmat3<-signif(summary(fitc)$coefficients,2) )
beta_se3=cbind(paste(Coefmat3[,1],"(",Coefmat3[,2] ,")",sep="") ,Coefmat3[,4])
rownames(beta_se3)=rownames(Coefmat3)

## cross validation
mspec.cv<-rep(NA,10)
msec.CV<-rep(NA,10)
nsamps=dim(df)[1]
pred.outcome<-rep(NA,nsamps)

for (kfold in 1:10) {
  idk=folds[[kfold]]
  fitK<-model_adaptive_LASSO_sel(pred=mat1[idk,Vars.subset],outcome=Y1[idk])
  msec.CV[kfold]=examine_mse(fitK)
  pred.obj<-predict(fitK, newdata=as.data.frame(mat1[-idk,]), se.fit = TRUE)
  pred.outcome[-idk]=pred.obj$fit
  
  rm(idk,fitK,pred.obj)
}

##c.v. performance measures;
mspec.CV<-sum((pred.outcome-Y1)^2)/length(Y1)

#########################################################
## ONE STEP Adaptive Lasso with pairwise interaction terms
#########################################################

## set up interaction terms;
#  First step: using .*. for all interactions
f <- as.formula(Y1 ~ .*.)
# Second step: using model.matrix to take advantage of f
matxy <- model.matrix(f, as.data.frame(mat1) )[,-1]

fit.inta<-model_adaptive_LASSO_sel(pred=matxy,outcome=Y1)
CoefInta<-as.data.frame(signif(summary(fit.inta)$coefficients,2))
beta_se4=cbind(paste(CoefInta[,1],"(",CoefInta[,2] ,")",sep="") ,CoefInta[,4])
rownames(beta_se4)=rownames(CoefInta)

#########################################################
## Chunk5: TWO STEP: CART + Adaptive Lasso with pairwise interaction terms
#########################################################

fcart <- as.formula(Y1 ~ .*.)
# Second step: using model.matrix to take advantage of f
matxyb <- model.matrix(fcart, as.data.frame(mat1[,sel.vars.bycart]) )[,-1]
fit.intb<-model_adaptive_LASSO_sel(pred=matxyb,outcome=Y1)

CoefIntb<-as.data.frame(signif(summary(fit.intb)$coefficients,2))
beta_se5=cbind(paste(CoefIntb[,1],"(",CoefIntb[,2] ,")",sep="") ,CoefIntb[,4])
rownames(beta_se5)=rownames(CoefIntb)


#########################################################
## Chunk6: TWO STEP: RF + Adaptive Lasso with pairwise interaction terms
#########################################################

RFVars.subset<- Vars.subset

fd <- as.formula(Y1 ~ .*.)
# Second step: using model.matrix to take advantage of f
mat2xy <- model.matrix(fd, as.data.frame(mat1[,RFVars.subset]) )[,-1]

fit.intc<-model_adaptive_LASSO_sel(pred=mat2xy,outcome=Y1)
# summary(fit.intc)$coefficients

CoefIntc<-as.data.frame(signif(summary(fit.intc)$coefficients,2))
beta_se6=cbind(paste(CoefIntc[,1],"(",CoefIntc[,2] ,")",sep="") ,CoefIntc[,4])
rownames(beta_se6)=rownames(CoefIntc)

######################################################################
###   Chunk 8. Performance calculation based on cross validation with interactions
####################################################################

mspe.cvinta<-rep(NA,10)
mse.cvinta<-rep(NA,10)
nsamps=dim(df)[1]
pred.outcome<-rep(NA,nsamps)

for (kfold in 1:10) {
  idk=folds[[kfold]]
  fmod <- as.formula(Y1 ~ .*.)
  subVars<-colnames(mat1)                   ## change for two step results;
  # Second step: using model.matrix to take advantage of f
  matxy.inter <- model.matrix(fmod, as.data.frame(mat1[,subVars]) )[,-1]
  fitK<-model_adaptive_LASSO_sel(pred=matxy.inter[idk,],outcome=Y1[idk])
  mse.cvinta[kfold]=examine_mse(fitK)
  pred.obj<-predict(fitK, newdata=as.data.frame(mat1[-idk,subVars]), se.fit = TRUE)
  pred.outcome[-idk]=pred.obj$fit
  
  rm(idk,fitK,pred.obj)
}

mspe.cvinta<-sum((pred.outcome-Y1)^2)/length(Y1)


######################################################################
###   Chunk 9. Performance calculation based on cross validation with interactions
### cart+aLasso
####################################################################


mspe.cvintb<-rep(NA,10)
mse.cvintb<-rep(NA,10)
nsamps=dim(df)[1]
pred.outcome<-rep(NA,nsamps)

for (kfold in 1:10) {
  idk=folds[[kfold]]
  fmod <- as.formula(Y1 ~ .*.)
  subVars<-sel.vars.bycart                 ## change for two step results;
  # Second step: using model.matrix to take advantage of f
  matxy.inter <- model.matrix(fmod, as.data.frame(mat1[,subVars]) )[,-1]
  fitK<-model_adaptive_LASSO_sel(pred=matxy.inter[idk,],outcome=Y1[idk])
  mse.cvintb[kfold]=examine_mse(fitK)
  pred.obj<-predict(fitK, newdata=as.data.frame(mat1[-idk,subVars]), se.fit = TRUE)
  pred.outcome[-idk]=pred.obj$fit
  
  rm(idk,fitK,pred.obj)
}

mspe.cvintb<-sum((pred.outcome-Y1)^2)/length(Y1)




######################################################################
###   Chunk 10. Performance calculation based on cross validation with interactions
### RF+aLasso
####################################################################


mspe.cvintc<-rep(NA,10)
mse.cvintc<-rep(NA,10)
nsamps=dim(df)[1]
pred.outcome<-rep(NA,nsamps)

for (kfold in 1:10) {
  idk=folds[[kfold]]
  fmod <- as.formula(Y1 ~ .*.)
  subVars<-RFVars.subset                 ## change for two step results;
  # Second step: using model.matrix to take advantage of f
  matxy.inter <- model.matrix(fmod, as.data.frame(mat1[,subVars]) )[,-1]
  fitK<-model_adaptive_LASSO_sel(pred=matxy.inter[idk,],outcome=Y1[idk])
  mse.cvintc[kfold]=examine_mse(fitK)
  pred.obj<-predict(fitK, newdata=as.data.frame(mat1[-idk,subVars]), se.fit = TRUE)
  pred.outcome[-idk]=pred.obj$fit
  
  rm(idk,fitK,pred.obj)
}

mspe.cvintc<-sum((pred.outcome-Y1)^2)/length(Y1)



#########################################################
## output 1 simu results;
#########################################################

library(rtf)
tim.str=Sys.time()


output<-paste("output/simu_results_",format(tim.str,"%d%b%y_%H%M%S"),".doc",collapse="")

rtf<-RTF(output,width=8.5,height=11,font.size=10,omi=c(1,1,1,1))
addHeader(rtf,title="Example simulation results",subtitle=paste(tim.str,"\n") )
addParagraph(rtf,paste("\n Simulation settings: \n effects=\n",strbeta,"\n") )
addParagraph(rtf,paste("\n correlation settings: \n rho between the first 15 vairables=",rho1,"\n",
                       "rho between the last 5 vairables=",rho2) )
addParagraph(rtf,paste("\n seed is set at:",rand.seed,"\n") )

addParagraph(rtf,"\n Table1. One step Adaptive Lasso,No Interaction terms, Results \n")
addTable(rtf,    beta_se1,
         font.size=10,row.names=TRUE,
         NA.string="-",col.justify="L",space.after=0.05)
addParagraph(rtf,sprintf("\n model1:multiple R squared=%.3f, adjusted R squared=%.3f ",
                         summary(fita)$r.squared,summary(fita)$adj.r.squared) )
addParagraph(rtf,sprintf("\n model1 performance:mse=%.3f, mse.cv=%.3f, mspe.cv=%.3f ",
                         msea,mean(msea.CV), mspea.CV) )

addParagraph(rtf,"\n Table2. Cart+Adaptive Lasso, No Interaction terms , Results \n")
addTable(rtf,    beta_se2,
         font.size=10,row.names=TRUE,
         NA.string="-",col.justify="L",space.after=0.05)
addParagraph(rtf,sprintf("\n model2:multiple R squared=%.3f, adjusted R squared=%.3f ",
                         summary(fitb)$r.squared,summary(fitb)$adj.r.squared) )
addParagraph(rtf,sprintf("\n model2 performance:mse=%.3f, mse.cv=%.3f, mspe.cv=%.3f ",
                         mseb,mean(mseb.CV), mspeb.CV) )



addParagraph(rtf,"\n Table3. RF+Adaptive Lasso, No Interaction terms , Results \n")
addTable(rtf,    beta_se3,
         font.size=10,row.names=TRUE,
         NA.string="-",col.justify="L",space.after=0.05)
addParagraph(rtf,sprintf("\n model3:multiple R squared=%.3f, adjusted R squared=%.3f ",
                         summary(fitc)$r.squared,summary(fitc)$adj.r.squared) )
addParagraph(rtf,sprintf("\n model3 performance:mse=%.3f, mse.cv=%.3f, mspe.cv=%.3f ",
                         examine_mse(fitc),mean(msec.CV), mspec.CV) )



addParagraph(rtf,"\n Table4. One step Adaptive Lasso, Pairwise Interaction terms, Results \n")
addTable(rtf,    beta_se4,
         font.size=10,row.names=TRUE,
         NA.string="-",col.justify="L",space.after=0.05)
addParagraph(rtf,sprintf("\n model4:multiple R squared=%.3f, adjusted R squared=%.3f ",
                         summary(fit.inta)$r.squared,summary(fit.inta)$adj.r.squared) )
addParagraph(rtf,sprintf("\n model4 performance:mse=%.3f, mse.cv=%.3f, mspe.cv=%.3f ",
                         examine_mse(fit.inta),mean(mse.cvinta), mspe.cvinta) )

addParagraph(rtf,"\n Table5. Cart+Adaptive Lasso, Pairwise Interaction terms , Results \n")
addTable(rtf,    beta_se5,
         font.size=10,row.names=TRUE,
         NA.string="-",col.justify="L",space.after=0.05)
addParagraph(rtf,sprintf("\n model5:multiple R squared=%.3f, adjusted R squared=%.3f ",
                         summary(fit.intb)$r.squared,summary(fit.intb)$adj.r.squared) )
addParagraph(rtf,sprintf("\n model5 performance:mse=%.3f, mse.cv=%.3f, mspe.cv=%.3f ",
                         examine_mse(fit.intb),mean(mse.cvintb), mspe.cvintb) )



addParagraph(rtf,"\n Table6. RF+Adaptive Lasso, Pairwise Interaction terms , Results \n")
addTable(rtf,    beta_se6,
         font.size=10,row.names=TRUE,
         NA.string="-",col.justify="L",space.after=0.05)
addParagraph(rtf,sprintf("\n model6:multiple R squared=%.3f, adjusted R squared=%.3f ",
                         summary(fit.intc)$r.squared,summary(fit.intc)$adj.r.squared) )
addParagraph(rtf,sprintf("\n model6 performance:mse=%.3f, mse.cv=%.3f, mspe.cv=%.3f ",
                         examine_mse(fit.intc),mean(mse.cvintc), mspe.cvintc) )
done(rtf)

save.image("output/simulation_results.Rdata")
#####################################################################################################
##############################ADD plots; 


library(forestplot)

## for plots;
fita
fitb
fitc
fit.inta
fit.intb
fit.intc

TrueBeta=c(0.2,	0.3,	-0.15,	0.2,	0.1,	-0.25,	0.1)
names(TrueBeta)=c("X1",	"X2",	"X9",	"X12",	"X15",	"X16",
                  "X1:X12")
## rename;
names(fita$coefficients)[-1]<-gsub("V","X",names(fita$coefficients)[-1])
names(fitb$coefficients)[-1]<-gsub("V","X",names(fitb$coefficients)[-1])
names(fitc$coefficients)[-1]<-gsub("V","X",names(fitc$coefficients)[-1])


## table text;
table3text= unique(c(
  names(coef(fita)[-1]) ,
  names(coef(fitb)[-1]),
  names(coef(fitc)[-1]),
  names(TrueBeta)) )

## row ids; 

idx.A<-match(table3text,names(coef(fita)))-1;idx.A[is.na(idx.A)]=length(table3text)+1
idx.B<-match(table3text,names(coef(fitb)))-1;idx.B[is.na(idx.B)]=length(table3text)+1
idx.C<-match(table3text,names(coef(fitc)))-1;idx.C[is.na(idx.C)]=length(table3text)+1
idx.T<-match(table3text, names(TrueBeta))
Reorder.Beta<-TrueBeta[idx.T];#Reorder.Beta[is.na(Reorder.Beta)]=NA

jpeg("Figure3.jpg", width = 1920, height = 1080,res=150)
forestplot(table3text, 
           legend = c("a.SingleStep(alasso)", "b.TwoStep(CART+alasso)","c.TwoStep(RF+alasso)","TrueBeta"),
           mean=cbind(fita$coefficients[-1][idx.A],fitb$coefficients[-1][idx.B],fitc$coefficients[-1][idx.C],Reorder.Beta),
           lower = cbind(confint(fita)[-1,1][idx.A],confint(fitb)[-1,1][idx.B],confint(fitc)[-1,1][idx.C],Reorder.Beta-0.001),
           upper = cbind(confint(fita)[-1,2][idx.A],confint(fitb)[-1,2][idx.B],confint(fitc)[-1,2][idx.C],Reorder.Beta+0.001 ),
           boxsize = .25, # We set the box size to better visualize the type
           line.margin = .15, # We need to add this to avoid crowding
           new_page = TRUE,
           col=fpColors(box=c("green","blue", "darkred","black")) 
)
dev.off()

########Figure 4 (With Interaction)
## rename;
names(fit.inta$coefficients)[-1]<-gsub("V","X",names(fit.inta$coefficients)[-1])
names(fit.intb$coefficients)[-1]<-gsub("V","X",names(fit.intb$coefficients)[-1])
names(fit.intc$coefficients)[-1]<-gsub("V","X",names(fit.intc$coefficients)[-1])



names(fit.intb$coefficients)[c(12,14,15,16)]=c("X14:X16","X20:X16","X3:X16","X3:X17")
      
table2text= unique(c(
  names(coef(fit.inta)[-1]) ,
  names(coef(fit.intb)[-1]),
  names(coef(fit.intc)[-1]),
  names(TrueBeta) ) ) 

table2text<-sort(table2text)
table2text<-table2text[c(1,21,31,9,15,17,3,setdiff(1:31,c(1,21,31,9,15,17,3) ))]

idx.rowA<-match(table2text,names(coef(fit.inta)))-1;idx.rowA[is.na(idx.rowA)]=length(table2text)+1
idx.rowB<-match(table2text,names(coef(fit.intb)))-1;idx.rowB[is.na(idx.rowB)]=length(table2text)+1
idx.rowC<-match(table2text,names(coef(fit.intc)))-1;idx.rowC[is.na(idx.rowC)]=length(table2text)+1

idx.rowT<-match(table2text, names(TrueBeta)); Reorder.Beta<-TrueBeta[idx.rowT];



jpeg("Figure4_interaction.jpg", width = 1920, height = 2160,res=150)

forestplot(table2text, 
           legend = c("a.SingleStep(alasso)", "b.TwoStep(CART+alasso)","c.TwoStep(RF+alasso)","TrueBeta"),
           mean=cbind(fit.inta$coefficients[-1][idx.rowA],fit.intb$coefficients[-1][idx.rowB],fit.intc$coefficients[-1][idx.rowC],Reorder.Beta),
           lower = cbind(confint(fit.inta)[-1,1][idx.rowA],confint(fit.intb)[-1,1][idx.rowB],confint(fit.intc)[-1,1][idx.rowC],Reorder.Beta-0.001),
           upper = cbind(confint(fit.inta)[-1,2][idx.rowA],confint(fit.intb)[-1,2][idx.rowB],confint(fit.intc)[-1,2][idx.rowC],Reorder.Beta+0.001),
           boxsize = .3, # We set the box size to better visualize the type
           line.margin = 0.15, # We need to add this to avoid crowding
           new_page = TRUE,
           col=fpColors(box=c("green","blue", "darkred","black"),
                        lines=c("green","blue", "darkred","black") ) 
)
dev.off()


