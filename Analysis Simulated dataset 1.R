rm(list=ls())
library(MASS)
library("xlsx")
library(nortest)
require(glmnet)
library(rtf)
library(randomForestSRC)
library(selectiveInference)
library(rms)
## set up environment;
source("Adaptive_LASSO_Script_exm.R")

rand.seed=sample.int(9999999,1)
set.seed(rand.seed)
## read in data;
datinp<-read.xlsx("dataset1.xlsx",sheetIndex = 1)
datinp<-datinp[,-1]
head(datinp)

## set up analysis;

df<-datinp
names(df)[1]<-"Y1"
df[,2:8]<-log(df[,2:8])

summary(lm(Y1~.,data=df))
mat1<-as.matrix(datinp[,-1])
head(mat1)
Y1=df$Y1
#########################################################
## ONE STEP Adaptive Lasso NO interaction
#########################################################
fita<-model_adaptive_LASSO_sel(pred=mat1,outcome=Y1)
summary(fita)$coefficients

ra<-(summary(fita))$residuals 

(msea<-sum(ra^2)/length(ra) )

#########################################################
## ONE STEP Adaptive Lasso with pairwise interaction terms
#########################################################

## set up interaction terms;
#  First step: using .*. for all interactions
f <- as.formula(Y1 ~ .*.)
# Second step: using model.matrix to take advantage of f
matxy <- model.matrix(f, as.data.frame(mat1) )[,-1]

fitb<-model_adaptive_LASSO_sel(pred=matxy,outcome=Y1)
summary(fitb)$coefficients

rb<-(summary(fitb))$residuals 
mseb<-sum(rb^2)/length(rb)

## try neural network;
library(neuralnet)
print(net.example1 <- neuralnet(Y1~X1+X2+X3+X4+X5+X6+X7+Z, df, hidden=1,
                              err.fct="sse", linear.output=TRUE, likelihood=TRUE))


main <- glm(Y1~X1+X2+X3+X4+X5+X6+X7+Z, df, family=gaussian())
full <- glm(Y1~X1*X2*X3*X4*X5*X6*X7*Z, df, family=gaussian())
pred1<-prediction(net.example1, list.glm=list(main=main, full=full))

ls(pred1)
outnn1<-cbind(df$Y1,pred1$glm.main[,"Y1"],pred1$glm.full[,"Y1"])
head(outnn1)
err1<-sum( (outnn1[,1]-outnn1[,2])^2)

plot(net.example1)

#########################################################
## Two step- RF first
#########################################################

## Step1- RF selection;
set.seed(123456789)
n.trees=5000
rf.obj <- rfsrc(Y1 ~ ., data = df,ntree =n.trees,tree.err=TRUE,importance=TRUE)
Vars.subset<-names(which(rf.obj$importance>0))
Vars.subset<-setdiff(Vars.subset,c("X6","X4"))
plot(rf.obj)


#########################################################
## Adaptive Lasso NO interaction terms
#########################################################

fitc<-model_adaptive_LASSO_sel(pred=mat1[,Vars.subset],outcome=Y1)
summary(fitc)$coefficients

#########################################################
## TWO STEP: Adaptive Lasso with pairwise interaction terms
#########################################################
fd <- as.formula(Y1 ~ .*.)
# Second step: using model.matrix to take advantage of f
mat2xy <- model.matrix(fd, as.data.frame(mat1[,Vars.subset]) )[,-1]

fitd<-model_adaptive_LASSO(pred=mat2xy,outcome=Y1)
summary(fitd)$coefficients

## output 1 simu results;
library(rtf)
tim.str=Sys.time()
output<-paste("output/simu_results_",format(tim.str,"%d%b%y_%H%M%S"),".doc",collapse="")

rtf<-RTF(output,width=8.5,height=11,font.size=10,omi=c(1,1,1,1))
addHeader(rtf,title="Example simulation results",subtitle=paste(tim.str,"\n") )
addParagraph(rtf,paste("\n Simulation settings: \n effects=\n",strbeta,"\n") )
addParagraph(rtf,paste("\n seed is set at:",rand.seed,"\n") )

addParagraph(rtf,"\n Table1. One step Adaptive Lasso,No Interaction terms, Results \n")
addTable(rtf,    signif(summary(fita)$coefficients,3),
         font.size=10,row.names=TRUE,
         NA.string="-",col.justify="L",space.after=0.05)
addParagraph(rtf,sprintf("\n model1:multiple R squared=%.3f, adjusted R squared=%.3f \n",
                         summary(fita)$r.squared,summary(fita)$adj.r.squared) )

addParagraph(rtf,"\n Table2. One step Adaptive Lasso, with pairwise Interaction terms, Results \n")
addTable(rtf,    signif(summary(fitb)$coefficients,3),
         font.size=10,row.names=TRUE,
         NA.string="-",col.justify="L",space.after=0.05)

addParagraph(rtf,sprintf("\n model2:multiple R squared=%.3f, adjusted R squared=%.3f \n",
                         summary(fitb)$r.squared,summary(fitb)$adj.r.squared) )
addParagraph(rtf,"\n Table3. Two step Approach, Random Forest followed by Adaptive Lasso,No Interaction terms, Results \n")
addTable(rtf,    signif(summary(fitc)$coefficients,3),
         font.size=10,row.names=TRUE,
         NA.string="-",col.justify="L",space.after=0.05)

addParagraph(rtf,sprintf("\n model3:multiple R squared=%.3f, adjusted R squared=%.3f \n",
                         summary(fitc)$r.squared,summary(fitc)$adj.r.squared) )

addParagraph(rtf,"\n Table4. Two step Approach, Random Forest followed by Adaptive Lasso, with Interaction terms 
             of the preselcted variables, Results \n")
addTable(rtf,    signif(summary(fitd)$coefficients,3),
         font.size=10,row.names=TRUE,
         NA.string="-",col.justify="L",space.after=0.05)

addParagraph(rtf,sprintf("\n model4:multiple R squared=%.3f, adjusted R squared=%.3f \n",
                         summary(fitd)$r.squared,summary(fitd)$adj.r.squared) )
done(rtf)

pdf(paste("output/RF_Plots_",format(tim.str,"%d%b%y_%H%M%S"),".pdf",collapse=""))
plot(rf.obj)
dev.off()