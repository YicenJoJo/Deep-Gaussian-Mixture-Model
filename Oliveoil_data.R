library('pdfCluster')
library('EMMIXskew')
library('aricode')
library('mclust')
library('cluster')
library('deepgmm')
library('MixGHD')
library('e1071')
library('ggplot2')
library('mltools')
library('data.table')
library('EMMIXmfa')
library('FactMixtAnalysis')
library('teigen')

ARI_results = rep(0,8)
M.R_results = rep(0,8)

data(oliveoil)
oliveoil$region <- as.factor(oliveoil$region)
newdata <- one_hot(as.data.table(oliveoil[,2:10]))    #One-hot encoding

a = which(oliveoil$macro.area=='South')
b = which(oliveoil$macro.area=='Sardinia')
c = which(oliveoil$macro.area=='Centre.North')
label = rep(0,length(oliveoil$macro))
label[a] = 1
label[b] = 2
label[c] = 3

range1 = factorial(3)
pp = permutations(3)



#The K means results
k = kmeans(newdata,3)
kmeans_results = k$cluster
a = which(kmeans_results==1)
b = which(kmeans_results==2)
c = which(kmeans_results==3)
c.r = rep(0,range1)
for (i in 1:range1){
  kmeans_results[a]=pp[i,1]
  kmeans_results[b]=pp[i,2]
  kmeans_results[c]=pp[i,3]
  tab <- table(label,kmeans_results)
  c.r[i] = (sum(diag(tab))/sum(tab))
} 
index = which(c.r==max(c.r))
kmeans_results[a]=pp[index,1]
kmeans_results[b]=pp[index,2]
kmeans_results[c]=pp[index,3]
mc.r = 1-max(c.r)
ARI_k = ARI(label,kmeans_results)
ARI_results[1]= ARI_k           #Record the results
M.R_results[1]= mc.r





#Mclust results
m = Mclust(newdata,3)          #Use VVV as default
Mclust_results = m$classification
a = which(Mclust_results==1)
b = which(Mclust_results==2)
c = which(Mclust_results==3)
c.r.m = rep(0,range1)
for (i in 1:range1){
  Mclust_results[a]=pp[i,1]
  Mclust_results[b]=pp[i,2]
  Mclust_results[c]=pp[i,3]
  tab <- table(label,Mclust_results)
  c.r.m[i] = (sum(diag(tab))/sum(tab))
} 
index = which(c.r.m==max(c.r.m))
Mclust_results[a]=pp[index,1]
Mclust_results[b]=pp[index,2]
Mclust_results[c]=pp[index,3]
ARI_m = ARI(label,Mclust_results)
mc.r.m = 1-max(c.r.m)
ARI_results[2]= ARI_m
M.R_results[2]= mc.r.m



#Hclust results
distance_data2 <- dist(newdata, method = 'euclidean')
h = hclust(distance_data2, method = "ward.D")
hclust_result <- cutree(h, k = 3)
a = which(hclust_result==1)
b = which(hclust_result==2)
c = which(hclust_result==3)
c.r.h = rep(0,range1)
for (i in 1:range1){
  hclust_result[a]=pp[i,1]
  hclust_result[b]=pp[i,2]
  hclust_result[c]=pp[i,3]
  tab <- table(label,hclust_result)
  c.r.h[i] = (sum(diag(tab))/sum(tab))
} 
index = which(c.r.h==max(c.r.h))
hclust_result[a]=pp[index,1]
hclust_result[b]=pp[index,2]
hclust_result[c]=pp[index,3]
ARI_h = ARI(label,hclust_result)
mc.r.h = 1-max(c.r.h)
ARI_results[3]= ARI_h
M.R_results[3]= mc.r.h






#PAM results
p = pam(newdata, 3)    #Partitioning Around Medoids
pam_results = p$clustering
a = which(pam_results==1)
b = which(pam_results==2)
c = which(pam_results==3)
c.r.p = rep(0,range1)
for (i in 1:range1){
  pam_results[a]=pp[i,1]
  pam_results[b]=pp[i,2]
  pam_results[c]=pp[i,3]
  tab <- table(label,pam_results)
  c.r.p[i] = (sum(diag(tab))/sum(tab))
} 
index = which(c.r.p==max(c.r.p))
pam_results[a]=pp[index,1]
pam_results[b]=pp[index,2]
pam_results[c]=pp[index,3]
ARI_p = ARI(label,pam_results)
mc.r.p = 1-max(c.r.p)
ARI_results[4]= ARI_p
M.R_results[4]= mc.r.p








#DGMM results
layers <- 2
k <- c(3,1)
r <- c(5,1)
it <- 100
eps <- 0.0001
dgm <-deepgmm(y = newdata, layers = layers, k = k, r = r,it = it, eps = eps) 
dgm_result = dgm$s[,1]

a = which(dgm_result==1)
b = which(dgm_result==2)
c = which(dgm_result==3)

c.r.dgm = rep(0,range1)
for (i in 1:range1){
  dgm_result[a]=pp[i,1]
  dgm_result[b]=pp[i,2]
  dgm_result[c]=pp[i,3]
  tab <- table(label,dgm_result)
  c.r.dgm[i] = (sum(diag(tab))/sum(tab))
} 

index = which(c.r.dgm==max(c.r.dgm))
dgm_result[a]=pp[index,1]
dgm_result[b]=pp[index,2]
dgm_result[c]=pp[index,3]
ARI_dgm = ARI(label,dgm_result)
mc.r.dgm = 1-max(c.r.dgm)
ARI_results[5]= ARI_dgm
M.R_results[5]= mc.r.dgm




 
# #MGHD results                             #Error: can not find estimated models
# mg = MGHD(newdata,G=3,modelSel='BIC' )
# mg_results = mg@map
# a = which(mg_results ==1)
# b = which(mg_results ==2)
# c = which(mg_results ==3)
# 
# c.r.mg = rep(0,range1)
# for (i in 1:range1){
#   mg_results [a]=pp[i,1]
#   mg_results [b]=pp[i,2]
#   mg_results [c]=pp[i,3]
#   tab <- table(label,mg_results )
#   c.r.mg[i] = (sum(diag(tab))/sum(tab))
# } 
# 
# index = which(c.r.mg==max(c.r.mg))
# mg_results [a]=pp[index,1]
# mg_results [b]=pp[index,2]
# mg_results [c]=pp[index,3]
# ARI_mg = ARI(label,mg_results ) 
# mc.r.mg = 1-max(c.r.mg)   
# ARI_results[6]= ARI_mg
# M.R_results[6]= mc.r.mg






#MFA results
mf = mfa(newdata,g=3,q=1,itmax=200, nkmeans=5, nrandom=5)
mf_results = mf$clust

a = which(mf_results ==1)
b = which(mf_results ==2)
c = which(mf_results ==3)

c.r.mf = rep(0,range1)
for (i in 1:range1){
  mf_results [a]=pp[i,1]
  mf_results [b]=pp[i,2]
  mf_results [c]=pp[i,3]
  tab <- table(label,mf_results )
  c.r.mf[i] = (sum(diag(tab))/sum(tab))
} 

index = which(c.r.mf==max(c.r.mf))
mf_results[a]=pp[index,1]
mf_results [b]=pp[index,2]
mf_results [c]=pp[index,3]
ARI_mf = ARI(label,mf_results ) 
mc.r.mf = 1-max(c.r.mf)     #Or can use misc() to obtain misclassification error
ARI_results[7]= ARI_mf
M.R_results[7]= mc.r.mf

#mixtures of multivariate t-distributions results                        #Error: can not find estimated models
# te = teigen(newdata, models="dfunconstrained", Gs=3,init = 'soft')
# te_results = te$classification
# a = which(te_results ==1)
# b = which(te_results ==2)
# c = which(te_results ==3)
# 
# c.r.te = rep(0,range1)
# for (i in 1:range1){
#   te_results [a]=pp[i,1]
#   te_results [b]=pp[i,2]
#   te_results [c]=pp[i,3]
#   tab <- table(label,te_results )
#   c.r.te[i] = (sum(diag(tab))/sum(tab))
# }
# 
# index = which(c.r.te==max(c.r.te))
# te_results[a]=pp[index,1]
# te_results [b]=pp[index,2]
# te_results [c]=pp[index,3]
# ARI_te = ARI(label,te_results )
# mc.r.te = 1-max(c.r.te)     #Or can use misc() to obtain misclassification error
# ARI_results[8]= ARI_te
# M.R_results[8]= mc.r.te


Name <- c("Kmeans", "Mclust", "Hclust", "PAM", "deepgmm","MGHD",'MFA','MMt')
ARI_results[6]= 'NAN'
M.R_results[6]= 'NAN'
ARI_results[8]= 'NAN'
M.R_results[8]= 'NAN'
ARI_Oliveoil_comparison <- data.frame(Name, ARI_results)
M.R__Oliveoil_comparison <-data.frame(Name, M.R_results)

print(ARI_Oliveoil_comparison)
print(M.R__Oliveoil_comparison)


