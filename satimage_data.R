library('EMMIXskew')
library('aricode')
library('mclust')
library('cluster')
library('deepgmm')
library('MixGHD')
library('e1071')
library('ggplot2')
library('EMMIXmfa')
library('FactMixtAnalysis')
library('teigen')


#Load the data
s<-read.csv("satimage.csv")
s_label = unique(s$class)
s_data = s[,-c(37)]  #The dataset used to be classified
sclass = s$class

ARI_results = rep(0,8)
M.R_results = rep(0,8)

label = rep(0,length(s$Aattr))   #The labels transformed into numerical form.
for (i in 1:length(s_label)) {
  a = which(sclass==s_label[i])
  label[a] = i
}

range1 = factorial(length(s_label)) #The all possible permutations
pp = permutations(length(s_label))






#The K means results
k = kmeans(s_data,length(s_label))
kmeans_results = k$cluster

x <- matrix(0,length(s_label),length(s$Aattr))   #Record the corresponding index of the clustering groups to test all permutations
indexx = rep(0,length(s_label))
for (i in 1:length(s_label)) {
  x[i,1:length(which(kmeans_results==i))] = which(kmeans_results==i) 
  indexx[i]=length(which(kmeans_results==i))}

c.r = rep(0,range1)
for (i in 1:range1){             #In each permutation, records the CR
  for (j in 1:length(s_label) ) {
    kmeans_results[x[j,1:indexx[j]]] = pp[i,j]
  } 
  tab <- table(label,kmeans_results)
  c.r[i] = (sum(diag(tab))/sum(tab))
}
index = which(c.r==max(c.r))    #Find the largest CR and use that permutation
mc.r = 1-max(c.r)
ARI_k = ARI(label,kmeans_results)
ARI_results[1]= ARI_k           #Record the results
M.R_results[1]= mc.r





#The MCLust results (GMM)
m = Mclust(s_data,length(s_label)) 
Mclust_results = m$classification
x <- matrix(0,length(s_label),length(s$Aattr))
indexx = rep(0,length(s_label))
for (i in 1:length(s_label)) {
  x[i,1:length(which(Mclust_results==i))] = which(Mclust_results==i) 
  indexx[i]=length(which(Mclust_results==i))}
c.r = rep(0,range1)
for (i in 1:range1){             #In each permatution, records the CR
  for (j in 1:length(s_label) ) {
    Mclust_results[x[j,1:indexx[j]]] = pp[i,j]
  } 
  tab <- table(label,Mclust_results)
  c.r[i] = (sum(diag(tab))/sum(tab))
}
index = which(c.r==max(c.r))
mc.r.m = 1-max(c.r)
ARI_m = ARI(label,Mclust_results)
ARI_results[2]= ARI_m
M.R_results[2]= mc.r.m

 






#The HcLust results
distance_data2 <- dist(s_data, method = 'euclidean')
h = hclust(distance_data2, method = "ward.D2")
hclust_result <- cutree(h, k = length(s_label))
x <- matrix(0,length(s_label),length(s$Aattr))
indexx = rep(0,length(s_label))
for (i in 1:length(s_label)) {
  x[i,1:length(which(hclust_result==i))] = which(hclust_result==i) 
  indexx[i]=length(which(hclust_result==i))}

c.r = rep(0,range1)
for (i in 1:range1){             #In each permatution, records the CR
  for (j in 1:length(s_label) ) {
    hclust_result[x[j,1:indexx[j]]] = pp[i,j]
  } 
  tab <- table(label,hclust_result)
  c.r[i] = (sum(diag(tab))/sum(tab))
}
index = which(c.r==max(c.r))
mc.r.h = 1-max(c.r)
ARI_h = ARI(label,hclust_result)
ARI_results[3]= ARI_h
M.R_results[3]= mc.r.h








#The PAM results
p = pam(s_data, length(s_label))    
pam_results = p$clustering
x <- matrix(0,length(s_label),length(s$Aattr))
indexx = rep(0,length(s_label))

for (i in 1:length(s_label)) {
  x[i,1:length(which(pam_results==i))] = which(pam_results==i) 
  indexx[i]=length(which(pam_results==i))}

c.r = rep(0,range1)
for (i in 1:range1){             #In each permatution, records the CR
  for (j in 1:length(s_label) ) {
    pam_results[x[j,1:indexx[j]]] = pp[i,j]
  } 
  tab <- table(label,pam_results)
  c.r[i] = (sum(diag(tab))/sum(tab))
}
index = which(c.r==max(c.r))
mc.r.p = 1-max(c.r)
ARI_p = ARI(label,pam_results)
ARI_results[4]= ARI_p
M.R_results[4]= mc.r.p










#The DGMM results
layers <- 3
k <- c(6,2,1)
r <- c(13,2,1)
it <- 100
eps <- 0.0001
dgm <-deepgmm(y = s_data, layers = layers, k = k, r = r,it = it, eps = eps,init = 'kmeans',init_est = "factanal") 
dgm_result = dgm$s[,1]
x <- matrix(0,length(s_label),length(s$Aattr))
indexx = rep(0,length(s_label))

for (i in 1:length(s_label)) {
  x[i,1:length(which(dgm_result ==i))] = which(dgm_result ==i) 
  indexx[i]=length(which(dgm_result ==i))}

c.r = rep(0,range1)
for (i in 1:range1){             #In each permatution, records the CR
  for (j in 1:length(s_label) ) {
    dgm_result [x[j,1:indexx[j]]] = pp[i,j]
  } 
  tab <- table(label,dgm_result )
  c.r[i] = (sum(diag(tab))/sum(tab))
}
index = which(c.r==max(c.r))
mc.r.dgm = 1-max(c.r)
ARI_dgm = ARI(label,dgm_result )
ARI_results[5]= ARI_dgm
M.R_results[5]= mc.r.dgm













#The MGHD results
mg = MGHD(s_data,G=length(s_label))
mg_results = mg@map
x <- matrix(0,length(s_label),length(s$Aattr))
indexx = rep(0,length(s_label))

for (i in 1:length(s_label)) {
  x[i,1:length(which(mg_results==i))] = which(mg_results==i) 
  indexx[i]=length(which(mg_results==i))}

c.r = rep(0,range1)
for (i in 1:range1){             #In each permatution, records the CR
  for (j in 1:length(s_label) ) {
    mg_results[x[j,1:indexx[j]]] = pp[i,j]
  } 
  tab <- table(label,mg_results)
  c.r[i] = (sum(diag(tab))/sum(tab))
}
index = which(c.r==max(c.r))
mc.r.mg = 1-max(c.r)
ARI_mg = ARI(label,mg_results)
ARI_results[6]= ARI_mg
M.R_results[6]= mc.r.mg












#The MFA results
mf = mfa(s_data,g=length(s_label),q=2,itmax=200, nkmeans=5, nrandom=5)
mf_results = mf$clust
x <- matrix(0,length(s_label),length(s$Aattr))
indexx = rep(0,length(s_label))

for (i in 1:length(s_label)) {
  x[i,1:length(which(mf_results==i))] = which(mf_results==i) 
  indexx[i]=length(which(mf_results==i))}

c.r = rep(0,range1)
for (i in 1:range1){             #In each permatution, records the CR
  for (j in 1:length(s_label) ) {
    mf_results[x[j,1:indexx[j]]] = pp[i,j]
  } 
  tab <- table(label,mf_results)
  c.r[i] = (sum(diag(tab))/sum(tab))
}

index = which(c.r==max(c.r))
mc.r.mf = 1-max(c.r)
ARI_mf = ARI(label,mf_results)
ARI_results[7]= ARI_mf
M.R_results[7]= mc.r.mf


#mixtures of multivariate t-distributions results
te = teigen(s_data, models="dfunconstrained", Gs=length(s_label),init = 'soft')
te_results = te$classification

x <- matrix(0,length(s_label),length(s$Aattr))
indexx = rep(0,length(s_label))

for (i in 1:length(s_label)) {
  x[i,1:length(which(te_results==i))] = which(te_results==i) 
  indexx[i]=length(which(te_results==i))}

c.r = rep(0,range1)
for (i in 1:range1){             #In each permatution, records the CR
  for (j in 1:length(s_label) ) {
    te_results[x[j,1:indexx[j]]] = pp[i,j]
  } 
  tab <- table(label,te_results)
  c.r[i] = (sum(diag(tab))/sum(tab))
}

index = which(c.r==max(c.r))
mc.r.te = 1-max(c.r)
ARI_te = ARI(label,te_results)
ARI_results[8]= ARI_te
M.R_results[8]= mc.r.te






Name <- c("Kmeans", "Mclust", "Hclust", "PAM", "deepgmm","MGHD",'MFA','MMt')
ARI_comparison <- data.frame(Name, ARI_results)
M.R_comparison <-data.frame(Name, M.R_results)

print(ARI_comparison)
print(M.R_comparison)
