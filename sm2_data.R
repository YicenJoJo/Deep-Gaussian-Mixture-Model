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


ARI_results = rep(0,7)
M.R_results = rep(0,7)
iter_number = 100
Name <- c("K", "M", "H", "PAM","dgm","MGHD",'MMt')
ms = list(c("random","factanal"),c("random","ppca"),c("hclass","factanal"),c("hclass","ppca"),c("kmeans","factanal"),c("kmeans","ppca"))
ff_ResultsA <- matrix(0,iter_number,7)
ff_ResultsA  = data.frame(ff_ResultsA)

ff_Resultsmr <- matrix(0,iter_number,7)
ff_Resultsmr   = data.frame(ff_Resultsmr )
names(ff_ResultsA) <- Name
names(ff_Resultsmr) <- Name

for (kk in 1:iter_number) {
  
  smiley_data = mlbench.smiley(n=1000, sd1 = 0.45, sd2 = 0.35)
  smiley_data2 = data.frame(smiley_data$x)
  smiley_data3 = cbind(smiley_data2,rnorm(1000,mean = 0,sd=0.5))
  smiley_label = smiley_data$classes
  
  range1 = factorial(4)
  pp = permutations(4)
  
  k = kmeans(smiley_data3,length(unique(smiley_label)))
  kmeans_results = k$cluster
  
  x <- matrix(0,length(unique(smiley_label)),length(smiley_data$classes))   #Record the corresponding index of the clustering groups to test all permutations
  indexx = rep(0,length(unique(smiley_label)))
  for (i in 1:length(unique(smiley_label))) {
    x[i,1:length(which(kmeans_results==i))] = which(kmeans_results==i) 
    indexx[i]=length(which(kmeans_results==i))}
  
  c.r = rep(0,range1)
  for (i in 1:range1){             #In each permutation, records the CR
    for (j in 1:length(unique(smiley_label)) ) {
      kmeans_results[x[j,1:indexx[j]]] = pp[i,j]
    } 
    tab <- table(smiley_label,kmeans_results)
    c.r[i] = (sum(diag(tab))/sum(tab))
  }
  index = which(c.r==max(c.r))    #Find the largest CR and use that permutation
  mc.r = 1-max(c.r)
  ARI_k = ARI(smiley_label,kmeans_results)
  ARI_results[1]= ARI_k           #Record the results
  M.R_results[1]= mc.r
  
  
  m = Mclust(smiley_data3,length(unique(smiley_label)))          #Use VVV as default
  Mclust_results = m$classification
  
  x <- matrix(0,length(unique(smiley_label)),length(smiley_data$classes))   #Record the corresponding index of the clustering groups to test all permutations
  indexx = rep(0,length(unique(smiley_label)))
  for (i in 1:length(unique(smiley_label))) {
    x[i,1:length(which(Mclust_results==i))] = which(Mclust_results==i) 
    indexx[i]=length(which(Mclust_results==i))}
  
  c.r = rep(0,range1)
  for (i in 1:range1){             #In each permutation, records the CR
    for (j in 1:length(unique(smiley_label)) ) {
      Mclust_results[x[j,1:indexx[j]]] = pp[i,j]
    } 
    tab <- table(smiley_label,Mclust_results)
    c.r[i] = (sum(diag(tab))/sum(tab))
  }
  index = which(c.r==max(c.r))    #Find the largest CR and use that permutation
  mc.r.m = 1-max(c.r)
  ARI_m = ARI(smiley_label,Mclust_results)
  ARI_results[2]= ARI_m           #Record the results
  M.R_results[2]= mc.r.m
  
  
  distance_smiley <- dist(smiley_data3, method = 'euclidean')
  h = hclust(distance_smiley, method = "ward.D2")
  hclust_result <- cutree(h, k = 4 )
  x <- matrix(0,length(unique(smiley_label)),length(smiley_data$classes))   #Record the corresponding index of the clustering groups to test all permutations
  indexx = rep(0,length(unique(smiley_label)))
  for (i in 1:length(unique(smiley_label))) {
    x[i,1:length(which(hclust_result==i))] = which(hclust_result==i) 
    indexx[i]=length(which(hclust_result==i))}
  
  c.r = rep(0,range1)
  for (i in 1:range1){             #In each permutation, records the CR
    for (j in 1:length(unique(smiley_label)) ) {
      hclust_result[x[j,1:indexx[j]]] = pp[i,j]
    } 
    tab <- table(smiley_label,hclust_result)
    c.r[i] = (sum(diag(tab))/sum(tab))
  }
  index = which(c.r==max(c.r))    #Find the largest CR and use that permutation
  mc.r.h = 1-max(c.r)
  ARI_h = ARI(smiley_label,hclust_result)
  ARI_results[3]= ARI_h           #Record the results
  M.R_results[3]= mc.r.h
  
  
  
  
  p = pam(smiley_data3, length(unique(smiley_label)))    #Partitioning Around Medoids
  pam_results = p$clustering
  x <- matrix(0,length(unique(smiley_label)),length(smiley_data$classes))   #Record the corresponding index of the clustering groups to test all permutations
  indexx = rep(0,length(unique(smiley_label)))
  for (i in 1:length(unique(smiley_label))) {
    x[i,1:length(which(pam_results==i))] = which(pam_results==i) 
    indexx[i]=length(which(pam_results==i))}
  
  c.r = rep(0,range1)
  for (i in 1:range1){             #In each permutation, records the CR
    for (j in 1:length(unique(smiley_label)) ) {
      pam_results[x[j,1:indexx[j]]] = pp[i,j]
    } 
    tab <- table(smiley_label,pam_results)
    c.r[i] = (sum(diag(tab))/sum(tab))
  }
  index = which(c.r==max(c.r))    #Find the largest CR and use that permutation
  mc.r.p = 1-max(c.r)
  ARI_p = ARI(smiley_label,pam_results)
  ARI_results[4]= ARI_p           #Record the results
  M.R_results[4]= mc.r.p
  
  
  # sn = EmSkew(smiley_data3, length(unique(smiley_label)),distr="msn")
  # sn_results = sn$clust
  # x <- matrix(0,length(unique(smiley_label)),length(smiley_data$classes))   #Record the corresponding index of the clustering groups to test all permutations
  # indexx = rep(0,length(unique(smiley_label)))
  # for (i in 1:length(unique(smiley_label))) {
  #   x[i,1:length(which(sn_results==i))] = which(sn_results==i) 
  #   indexx[i]=length(which(sn_results==i))}
  # 
  # c.r = rep(0,range1)
  # for (i in 1:range1){             #In each permutation, records the CR
  #   for (j in 1:length(unique(smiley_label)) ) {
  #     sn_results[x[j,1:indexx[j]]] = pp[i,j]
  #   } 
  #   tab <- table(smiley_label,sn_results)
  #   c.r[i] = (sum(diag(tab))/sum(tab))
  # }
  # index = which(c.r==max(c.r))    #Find the largest CR and use that permutation
  # mc.r.sn = 1-max(c.r)
  # ARI_sn = ARI(smiley_label,sn$clust)
  # ARI_results[5]= ARI_sn           #Record the results
  # M.R_results[5]= mc.r.sn
  
  bic_value = rep(0,5)
  layers <- 2
  r <- c(2,1)
  it <- 100
  eps <- 0.001 
  k <- c(4,1)
  for (j in 1:5) { 
    dgm <-deepgmm(y = smiley_data3, layers = layers, k = k, r = r,it = it, eps = eps,init = ms[[j]][1],init_est = ms[[j]][2]) 
    bic_value[j] = dgm$bic
  }
  
  
  dgm <-deepgmm(y = smiley_data3, layers = layers, k = k, r = r,it = it, eps = eps,init = ms[[which(bic_value==min(bic_value))]][1],init_est = ms[[which(bic_value==min(bic_value))]][2]) 
  dgm_result = dgm$s[,1]
  
  
  x <- matrix(0,length(unique(smiley_label)),length(smiley_data$classes))   #Record the corresponding index of the clustering groups to test all permutations
  indexx = rep(0,length(unique(smiley_label)))
  for (i in 1:length(unique(smiley_label))) {
    x[i,1:length(which(dgm_result==i))] = which(dgm_result==i) 
    indexx[i]=length(which(dgm_result==i))}
  
  c.r = rep(0,range1)
  for (i in 1:range1){             #In each permutation, records the CR
    for (j in 1:length(unique(smiley_label)) ) {
      dgm_result[x[j,1:indexx[j]]] = pp[i,j]
    } 
    tab <- table(smiley_label,dgm_result)
    c.r[i] = (sum(diag(tab))/sum(tab))
  }
  index = which(c.r==max(c.r))    #Find the largest CR and use that permutation
  mc.r.dgm = 1-max(c.r)
  ARI_dgm = ARI(smiley_label,dgm_result)
  ARI_results[5]= ARI_dgm           #Record the results
  M.R_results[5]= mc.r.dgm
  
  
  
  
  mg = MGHD(smiley_data3,G=length(unique(smiley_label)))
  mg_results = mg@map
  x <- matrix(0,length(unique(smiley_label)),length(smiley_data$classes))   #Record the corresponding index of the clustering groups to test all permutations
  indexx = rep(0,length(unique(smiley_label)))
  for (i in 1:length(unique(smiley_label))) {
    x[i,1:length(which(mg_results==i))] = which(mg_results==i) 
    indexx[i]=length(which(mg_results==i))}
  
  c.r = rep(0,range1)
  for (i in 1:range1){             #In each permutation, records the CR
    for (j in 1:length(unique(smiley_label)) ) {
      mg_results[x[j,1:indexx[j]]] = pp[i,j]
    } 
    tab <- table(smiley_label,mg_results)
    c.r[i] = (sum(diag(tab))/sum(tab))
  }
  index = which(c.r==max(c.r))    #Find the largest CR and use that permutation
  mc.r.mg = 1-max(c.r)
  ARI_mg = ARI(smiley_label,mg_results)
  ARI_results[6]= ARI_mg           #Record the results
  M.R_results[6]= mc.r.mg
  
  ff_ResultsA[kk,] = ARI_results
  ff_Resultsmr[kk,] = M.R_results 
  
  te = teigen(smiley_data3, models="dfunconstrained", Gs=length(unique(smiley_label)))
  te_results = te$classification
  x <- matrix(0,length(unique(smiley_label)),length(smiley_data$classes))   #Record the corresponding index of the clustering groups to test all permutations
  indexx = rep(0,length(unique(smiley_label)))
  for (i in 1:length(unique(smiley_label))) {
    x[i,1:length(which(te_results==i))] = which(te_results==i) 
    indexx[i]=length(which(te_results==i))}
  
  c.r = rep(0,range1)
  for (i in 1:range1){             #In each permutation, records the CR
    for (j in 1:length(unique(smiley_label)) ) {
      te_results[x[j,1:indexx[j]]] = pp[i,j]
    } 
    tab <- table(smiley_label,te_results)
    c.r[i] = (sum(diag(tab))/sum(tab))
  }
  index = which(c.r==max(c.r))    #Find the largest CR and use that permutation
  mc.r.te = 1-max(c.r)
  ARI_te = ARI(smiley_label,te_results)
  ARI_results[7]= ARI_te           #Record the results
  M.R_results[7]= mc.r.te
  
  ff_ResultsA[kk,] = ARI_results
  ff_Resultsmr[kk,] = M.R_results 
  
  
  
  
  
  
  }

print(ff_ResultsA)
print(ff_Resultsmr)
boxplot(ff_ResultsA[1:7],names=Name,outline=FALSE,xlab='Models',ylab = 'ARI')
boxplot(ff_Resultsmr[1:7],names=Name,outline=FALSE,xlab='Models',ylab = 'Misclassification Rate')

ARI_mean = apply(ff_ResultsA,2,mean)
ARI_sd = apply(ff_ResultsA,2,sd)

M.r_mean = apply(ff_Resultsmr,2,mean)
M.r_sd = apply(ff_Resultsmr,2,sd)