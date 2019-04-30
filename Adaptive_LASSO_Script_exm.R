model_adaptive_LASSO_sel<-function(pred,outcome,lamda.sel="min"){
##set.seed(87654321)
  Y=outcome  ## vector n by 1
  XX=pred    ## matrix n by m
  ##colnames(XX)=paste("V",1:20,sep = "")
## ridge regression to get the wts;
gamma=1
cv.ridge <- cv.glmnet(XX, Y, family='gaussian', alpha=0, standardize=FALSE)
w3 <- 1/abs(matrix(coef(cv.ridge, s=cv.ridge$lambda.min)
                   [, 1][2:(ncol(XX)+1)] ))^1 ## Using gamma = 1
w3[w3[,1] == Inf] <- 999999999 ## Replacing values estimated as Infinite for 999999999

## Adaptive Lasso
cv.lasso <- cv.glmnet(XX, Y, family='gaussian', alpha=1, parallel=FALSE, 
                      standardize=FALSE, penalty.factor=w3)
if (lamda.sel=="min") bcoef <- coef(cv.lasso, s='lambda.min')
if (lamda.sel=="1se") bcoef <- coef(cv.lasso, s='lambda.1se')

selected_attributes <- (bcoef@i[-1]) ## Considering the structure of the data frame dataF as shown earlier
selected_metals<-colnames(XX)[selected_attributes]

## Final glm model;
datFrame=data.frame(cbind(Y,XX) )
fitb<-lm(as.formula(sprintf("Y~%s",paste(selected_metals,collapse="+")) ),
         data=datFrame)
#selb<-round(summary(fitb)$coefficients,4)
return(fitb)
}
