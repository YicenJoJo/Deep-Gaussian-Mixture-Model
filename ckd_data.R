library('mlbench')
library('EMMIXskew')
library('aricode')
library('mclust')
library('cluster')
library('deepgmm')
library('MixGHD')
library('e1071')
library('ggplot2')
library('teigen')

# Load the data ---- Chronic Kidney Disease data

# This data frame contains 203 rows (observations) and 13 columns (variables): 1.) ckdclass: 
# There  2 classes, ckd or notckd 
# 2.) age: in years 
# 3.) blood.pressure: in mm/Hg 
# 4.) blood.glucose.random:in mgs/dl 
# 5.) blood.urea: in mgs/dl 
# 6.) serum.creatinine: in mgs/dl 
# 7.) sodium: in mEq/L 
# 8.) potassium: in mEq/L 
# 9.) hemoglobin: in gms 
# 10.) packed.cell.volume 
# 11.) white.blood.cell.count: in cells/cmm 
# 12.) red.blood.cell.count: in cells/cmm

data(ckd)               # contained in teigen-package
ckd_label = unique(ckd$ckdmem)
ckd_data = ckd[,-c(1)]  #The dataset used to be classified
ckdclass = ckd$ckdmem

ARI_results = rep(0,8)
M.R_results = rep(0,8)

label = rep(0,length(ckd$age))   #The labels transformed into numerical form.
for (i in 1:length(ckd_label)) {
  a = which(ckdclass==ckd_label[i])
  label[a] = i
}

range1 = factorial(length(ckd_label)) #The all possible permutations
pp = permutations(length(ckd_label))




#The K means results
k = kmeans(ckd_data,length(ckd_label))
kmeans_results = k$cluster

x <- matrix(0,length(ckd_label),length(ckd$age))   #Record the corresponding index of the clustering groups to test all permutations
indexx = rep(0,length(ckd_label))
for (i in 1:length(ckd_label)) {
  x[i,1:length(which(kmeans_results==i))] = which(kmeans_results==i) 
  indexx[i]=length(which(kmeans_results==i))}

c.r = rep(0,range1)
for (i in 1:range1){             #In each permutation, records the CR
  for (j in 1:length(ckd_label) ) {
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
m = Mclust(ckd_data,length(ckd_label)) 
Mclust_results = m$classification
x <- matrix(0,length(ckd_label),length(ckd$age))   
indexx = rep(0,length(ckd_label))
for (i in 1:length(ckd_label)) {
  x[i,1:length(which(Mclust_results==i))] = which(Mclust_results==i) 
  indexx[i]=length(which(Mclust_results==i))}

c.r = rep(0,range1)
for (i in 1:range1){            
  for (j in 1:length(ckd_label) ) {
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
distance_data2 <- dist(ckd_data, method = 'euclidean')
h = hclust(distance_data2, method = "ward.D2")
hclust_result <- cutree(h, k = length(ckd_label))
x <- matrix(0,length(ckd_label),length(ckd$age))   
indexx = rep(0,length(ckd_label))
for (i in 1:length(ckd_label)) {
  x[i,1:length(which(hclust_result==i))] = which(hclust_result==i) 
  indexx[i]=length(which(hclust_result==i))}

c.r = rep(0,range1)
for (i in 1:range1){            
  for (j in 1:length(ckd_label) ) {
    hclust_result[x[j,1:indexx[j]]] = pp[i,j]
  } 
  tab <- table(label,hclust_result)
  c.r[i] = (sum(diag(tab))/sum(tab))
}
index = which(c.r==max(c.r))    
ARI_h = ARI(label,hclust_result)
ARI_results[3]= ARI_h           
M.R_results[3]= mc.r.h



#The PAM results
p = pam(ckd_data, length(ckd_label))    
pam_results = p$clustering
x <- matrix(0,length(ckd_label),length(ckd$age))   
indexx = rep(0,length(ckd_label))
for (i in 1:length(ckd_label)) {
  x[i,1:length(which(pam_results==i))] = which(pam_results==i) 
  indexx[i]=length(which(pam_results==i))}

c.r = rep(0,range1)
for (i in 1:range1){             
  for (j in 1:length(ckd_label) ) {
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


#The DGMM results                                 #The best model I found is r <- c(5,1), k <- c(2,3)
bic_value = rep(0,5)
layers <- 2
r <- c(5,1)
it <- 100
eps <- 0.001 
for (j in 1:5) { 
  k <- c(2,j)
  dgm <-deepgmm(y = ckd_data, layers = layers, k = k, r = r,it = it, eps = eps) 
  bic_value[j] = dgm$bic
}
k <- c(2,which(bic_value==min(bic_value)))
dgm <-deepgmm(y = ckd_data, layers = layers, k = k, r = r,it = it, eps = eps) 
dgm_result = dgm$s[,1]

x <- matrix(0,length(ckd_label),length(ckd$age))  
indexx = rep(0,length(ckd_label))
for (i in 1:length(ckd_label)) {
  x[i,1:length(which(dgm_result==i))] = which(dgm_result==i) 
  indexx[i]=length(which(dgm_result==i))}

c.r = rep(0,range1)
for (i in 1:range1){             
  for (j in 1:length(ckd_label) ) {
    dgm_result[x[j,1:indexx[j]]] = pp[i,j]
  } 
  tab <- table(label,dgm_result)
  c.r[i] = (sum(diag(tab))/sum(tab))
}
index = which(c.r==max(c.r))    
mc.r.dgm = 1-max(c.r)
ARI_dgm = ARI(label,dgm_result)
ARI_results[5]= ARI_dgm           
M.R_results[5]= mc.r.dgm 



#The MGHD results
mg = MGHD(ckd_data,G=length(ckd_label))
mg_results = mg@map

x <- matrix(0,length(ckd_label),length(ckd$age))   
indexx = rep(0,length(ckd_label))
for (i in 1:length(ckd_label)) {
  x[i,1:length(which(mg_results==i))] = which(mg_results==i) 
  indexx[i]=length(which(mg_results==i))}

c.r = rep(0,range1)
for (i in 1:range1){             
  for (j in 1:length(ckd_label) ) {
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
mf = mfa(ckd_data,g=length(ckd_label),q=2,itmax=200, nkmeans=5, nrandom=5)
mf_results = mf$clust
x <- matrix(0,length(ckd_label),length(ckd$age))   
indexx = rep(0,length(ckd_label))
for (i in 1:length(ckd_label)) {
  x[i,1:length(which(mf_results==i))] = which(mf_results==i) 
  indexx[i]=length(which(mf_results==i))}

c.r = rep(0,range1)
for (i in 1:range1){             
  for (j in 1:length(ckd_label) ) {
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
te = teigen(ckd_data, models="dfunconstrained", Gs=length(ckd_label))
te_results = te$classification
x <- matrix(0,length(ckd_label),length(ckd$age))   
indexx = rep(0,length(ckd_label))
for (i in 1:length(ckd_label)) {
  x[i,1:length(which(te_results==i))] = which(te_results==i) 
  indexx[i]=length(which(te_results==i))}

c.r = rep(0,range1)
for (i in 1:range1){            
  for (j in 1:length(ckd_label) ) {
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
ARI_ckd_comparison <- data.frame(Name, ARI_results)
M.R_ckd_comparison <-data.frame(Name, M.R_results)

print(ARI_ckd_comparison)
print(M.R_ckd_comparison)
