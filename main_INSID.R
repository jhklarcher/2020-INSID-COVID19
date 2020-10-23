rm(list = ls())
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

#Save several directories
BaseDir       <- getwd()
CodesDir      <- paste(BaseDir, "Codes", sep="/")
FiguresDir    <- paste(BaseDir, "Figures", sep="/")
ResultsDir    <- paste(BaseDir, "Results", sep="/")
DataDir       <- paste(BaseDir, "dados",sep="/")

#Load Packages
setwd(CodesDir)
source("checkpackages.R")
source("elm_caret.R")
source("Metricas.R")


packages<-c("forecast","cowplot","Metrics","caret","elmNNRcpp","tcltk","TTR",
            "foreach","iterators","doParallel","lmtest","httr","jsonlite","magrittr",
            "xlsx","caretEnsemble")

sapply(packages,packs)

library(extrafont)
windowsFonts(Times = windowsFont("TT Times New Roman"))
library(ggplot2)
library(Cairo)

#Checa quantos núcleos existem

ncl<-detectCores();ncl

#Registra os clusters a serem utilizados
cl <- makeCluster(ncl-1);registerDoParallel(cl)
#########################################################################################
setwd(DataDir)

data       <-read.table(file="covid-municipios-ENSID.csv",header=TRUE,sep=";",dec=",")

data       <-data[,-2]   #Removendo a Localidade
data       <-split(data,data$city)
City       <-c("Brasilia","Fortaleza","Rio de Janeiro","Sao Paulo", "Salvador")#,"Cascavel","Francisco Beltrão","Maringá","Pato Branco")
#data       <-data[City]
#names(data)<-City

#lapply(data,function(x){apply(x$casosAcumulado,2,max)})

#Dates for city
for(i in 1:length(data))
{
  n                 <-dim(data[[i]])[1]
  data[[i]][,'date']<-as.Date(seq(Sys.Date()-(n+2), by = "day", length.out = n))
}

#Statistics

descriptives<-lapply(data,function(x){apply(x[,c('cum_confirmed','new_confirmed')],2,basicStats)})
names_desc<-names(descriptives)
desc_summary<-list()
for(i in 1:length(descriptives))
  desc_summary[[i]]<-do.call(cbind.data.frame,descriptives[[i]])

desc_final<-do.call(cbind.data.frame,desc_summary)

write.xlsx(desc_final,file="Descriptive_Measures.xlsx")


#Objetos para treinamento
data_m  <-list();  #Objeto para receber os dados para cada treinamento
Models  <-list();  #Objeto para receber os modelos treinados
Arimas  <-list();
Params  <-list();  #Recebe os parâmetros
models  <-c("svmLinear2","brnn","knn","treebag") #Modelos a serem usados
k                 <-1        #Contador
a                 <-1
Aux_1             <-matrix(rep(0,times=25),nrow=5,ncol=5,byrow=TRUE)
Aux_2             <-rep(0,times=5)
Ptrain            <-list();               Ptest<-list();
Etrain            <-list();               Etest<-list();
Metrics           <-matrix(nrow=6,ncol=4)
colnames(Metrics) <-c("RMSE","MAE","SMAPE","SD")
row.names(Metrics)<-c("SVR","BRNN","KNN","BAGGING","Stack","ARIMA")
Metrics_City<-list()
horizontes<-c(1,3,6)

#########################Modeling#####################
cat("\014")

for(H in 1:length(horizontes)) #Variando o Horizonte de previsão
{
  Horizon<-horizontes[H]
  
  for(s in 1:length(data))           #Variando o conjunto de dados
  {
    {
      Data_state      <-sort(c(Aux_2,as.vector(data[[s]]$cum_confirmed)))
      #--------------------------Construindo Conjuno-----------------------------#
      data_m<-lags(Data_state, n = 5) #n representa o número de lags
      
      colnames(data_m)<-c("cum_confirmed",paste("Lag",1:5,sep=""))
      
      #----------------------Divisão em treinamento e teste---------------------#
      n      <-dim(data_m)[1]    #Número de observações
      cut    <-n-6               #Ponto de corte entre treino e teste
      
      
      ptrain          <-matrix(nrow=cut,ncol=length(models)+2); #Recebe predições 3SA das componentes
      ptest           <-matrix(nrow=n-cut,ncol=length(models)+2);  #Recebe predições 3SA das componentes
      colnames(ptrain)<-c(models,"Cubist-Stack","ARIMA");colnames(ptest) <-c(models,"Cubist-Stack","ARIMA")
      
      
      errorstrain           <-matrix(nrow=cut,ncol=length(models)+2); #Recebe predições 3SA das componentes
      errorstest            <-matrix(nrow=n-cut,ncol=length(models)+2);  #Recebe predições 3SA das componentes
      colnames(errorstrain )<-c(models,"Cubist-Stack","ARIMA");colnames(errorstest) <-c(models,"Cubist-Stack","ARIMA")
      
      fitControl2<- trainControl(method= "cv",number=5,savePredictions="final") 
      
      #Train and Test
      train  <-data_m[1:cut,];
      Y_train<-train[,1];X_train<-train[,-1]
      test   <-tail(data_m,n-cut)
      Y_test <-test[,1]; X_test <-test[,-1]
      
      #Objects for Out-Of-Sample forecasting
      Y_train_OOS<-data_m[,1]
      X_train_OOS<-data_m[,-1]
      
      #----------------------Divisões para Treinamento----------------------------#
      
    }
    
    #-----------------------Training---------------------------#
    for(i in 1:(length(models)+2)) #Aqui está indo de 1 até número de modelos +1 pois tem o stacking
    {
      options(warn=-1)
      if(i != 6)
      {
        if(i != 5)
        {
          set.seed(1234)
          Models[[k]]<-train(as.data.frame(X_train[1:cut,]),as.vector(Y_train),
                             method=models[[i]],
                             preProcess = c("center","scale"), #Processamento Centro e Escala
                             tuneLength= 4,                    #Número de tipos de parâmetros 
                             trControl = fitControl2,verbose=FALSE)
          
          
        }
        else 
        {
          
          #Stacking ensemble
          modelst <- caretList(as.data.frame(X_train[1:cut,]),as.vector(Y_train),
                               trControl=fitControl2, 
                               preProcess = c("center","scale"),
                               methodList=models)
          Models[[k]] <-caretStack(modelst, 
                                   trControl=fitControl2, 
                                   method='cubist',
                                   preProcess = c("center","scale"),
                                   tuneLength= 5)
          
        }
        
        #-----------------------Salvando Parâmetros--------------------------------#
        Params[[k]]<-Models[[k]]$bestTune
        
      }
      else
      {
        Arimas[[a]]<-auto.arima(Y_train)
      }
      
      
      #---------------------------Lags-Names-----------------------------------------#
      Lag1<-match("Lag1",colnames(X_test));Lag2<-match("Lag2",colnames(X_test))
      Lag3<-match("Lag3",colnames(X_test));Lag4<-match("Lag4",colnames(X_test))
      Lag5<-match("Lag5",colnames(X_test))
      #------------------Recursive Forecasting for train and test sets-----------
      #Aqui, o conjunto de análise é dividido em n conjuntos de h observações. Nesse caso
      #A cada 3 predições, é reiniciado o processo de previsão. Caso isso não seja feito,
      #as predições continuarão a ser atualizadas e o erro carregado.
      #Se desejar h>3, basta tirar fazer H<-HORIZONTE DESEJADO e descomentar  
      #X_trainm[p+3,Lag3]<-ptrain[p,m] e #X_testm[p+3,Lag3]<-ptest[p,m]
      
      if(i != 6)
      {
        #Vai colocar as predições em cada coluna
        if(Horizon==1)
        {
          h<-Horizon
          
          #Train
          ptrain[1:cut,i]<-round(predict(Models[[k]], X_train[1:cut,]))
          #Test 
          ptest[1:6,i]<-round(predict(Models[[k]],X_test[1:6,]))
          
        }
        else if(Horizon==3)
        {
          h<-Horizon
          #Treinando e prevendo cada componente com cada modelo
          
          X_trainm<-rbind(train[,-1],Aux_1);
          X_testm <-rbind(test[,-1],Aux_1);
          colnames(X_trainm)=colnames(X_train)
          colnames(X_testm) =colnames(X_test)
          #Train
          for(p in 1:cut)
          {
            if(p%%h !=1) #Sempre reinicia na divisão de resto 1-->Multiplos de h+1
            {
              ptrain[p,i]<-round(predict(Models[[k]], as.data.frame(t(X_trainm[p,]))))
              X_trainm[p+1,Lag1]<-ptrain[p,i]
              X_trainm[p+2,Lag2]<-ptrain[p,i]
              
            }
            else
            {
              X_trainm[p:(n-cut),]<-X_train[p:(n-cut),]
              ptrain[p,i]       <-round(predict(Models[[k]], as.data.frame(t(X_trainm[p,]))))
              X_trainm[p+1,Lag1]<-ptrain[p,i]
              X_trainm[p+2,Lag2]<-ptrain[p,i]
              
            }
          }
          #Test  
          for(p in 1:(n-cut))
          {
            if(p%%h !=1)
            {
              ptest[p,i]<-round(predict(Models[[k]], as.data.frame(t(X_testm[p,]))))
              X_testm[p+1,Lag1]<-ptest[p,i]
              X_testm[p+2,Lag2]<-ptest[p,i]
              
              
            }
            else
            {
              X_testm[p:(n-cut),]<-X_test[p:(n-cut),]
              ptest[p,i]       <-round(predict(Models[[k]], as.data.frame(t(X_testm[p,]))))
              X_testm[p+1,Lag1]<-ptest[p,i]
              X_testm[p+2,Lag2]<-ptest[p,i]
              
            }
          }
        }
        else
        {
          #Treinando e prevendo cada componente com cada modelo
          
          X_trainm<-rbind(train[,-1],Aux_1);
          X_testm <-rbind(test[,-1],Aux_1);
          colnames(X_trainm)=colnames(X_train)
          colnames(X_testm) =colnames(X_test)
          h<-Horizon
          
          #Train
          for(p in 1:cut)
          {
            if(p%%h !=1) #Sempre reinicia na divisão de resto 1-->Multiplos de h+1
            {
              ptrain[p,i]<-round(predict(Models[[k]], as.data.frame(t(X_trainm[p,]))))
              X_trainm[p+1,Lag1]<-ptrain[p,i]
              X_trainm[p+2,Lag2]<-ptrain[p,i]
              X_trainm[p+3,Lag3]<-ptrain[p,i]
              X_trainm[p+4,Lag4]<-ptrain[p,i]
              X_trainm[p+5,Lag5]<-ptrain[p,i]
              
            }
            else
            {
              X_trainm[p:(n-cut),]<-X_train[p:(n-cut),]
              ptrain[p,i]       <-round(predict(Models[[k]], as.data.frame(t(X_trainm[p,]))))
              X_trainm[p+1,Lag1]<-ptrain[p,i]
              X_trainm[p+2,Lag2]<-ptrain[p,i]
              X_trainm[p+3,Lag3]<-ptrain[p,i]
              X_trainm[p+4,Lag4]<-ptrain[p,i]
              X_trainm[p+5,Lag5]<-ptrain[p,i]
              
            }
          }
          #Test  
          for(p in 1:(n-cut))
          {
            if(p%%h !=1)
            {
              ptest[p,i]<-round(predict(Models[[k]], as.data.frame(t(X_testm[p,]))))
              X_testm[p+1,Lag1]<-ptest[p,i]
              X_testm[p+2,Lag2]<-ptest[p,i]
              X_testm[p+3,Lag3]<-ptest[p,i]
              X_testm[p+4,Lag4]<-ptest[p,i]
              X_testm[p+5,Lag5]<-ptest[p,i]
              
            }
            else
            {
              X_testm[p:(n-cut),]<-X_test[p:(n-cut),]
              ptest[p,i]       <-round(predict(Models[[k]], as.data.frame(t(X_testm[p,]))))
              X_testm[p+1,Lag1]<-ptest[p,i]
              X_testm[p+2,Lag2]<-ptest[p,i]
              X_testm[p+3,Lag3]<-ptest[p,i]
              X_testm[p+4,Lag4]<-ptest[p,i]
              X_testm[p+5,Lag5]<-ptest[p,i]
            }
          }
        }
        
      }
      else
      {
        predictions_arima<-forecast::forecast(Arimas[[a]], h=6)
        ptest[,i] <-round(c(predictions_arima$mean))
        ptrain[,i] <-round(c(Arimas[[a]]$fitted))
        
      }
      
      #Erros
      errorstest[,i] <-round(Y_test-ptest[1:6,i])
      errorstrain[,i] <-round(Y_train-ptrain[1:cut,i])
      
      
      #Metrics in test
      criterias<-PM(Y_test,ptest[1:6,i],mean(Y_test))
      Metrics[i,]<-c(criterias[,c("RMSE","MAE","SMAPE")],sd(errorstest[1:6,i]))
      
      #Cada elemento da lista recebera uma combinação de process com control
      
      cat("City:",City[s],"Horizon:",Horizon,
          sprintf('RMSE: %0.3f',Metrics[i,1]),
          sprintf('SMAPE: %0.3f',Metrics[i,3]),
          sprintf('MAE: %0.3f'  ,Metrics[i,2]),
          sprintf('SD: %0.3f'   ,Metrics[i,4]),
          'Model:',ifelse(i==6,"Arima",ifelse(i==5,"Stack-Cubist",models[i])),
          "\n")
      
      k<-k+1
    }
    a<-a+1
    
    Metrics_City[[s]]  <-Metrics[order(Metrics[,3],decreasing=FALSE),]
    Ptrain[[s]]          <-ptrain;         
    Ptest[[s]]           <-ptest;
    Etrain[[s]]          <-errorstrain;
    Etest[[s]]           <-errorstest;
  }
  
  names(Metrics_City)<-City
  
  Results<-list(Ptrain,Ptest,Etest,Etrain,Params,Models,Arimas,Metrics_City)
  
  name   <-paste("Results_",Horizon,"SA_",Sys.Date(),".RData", sep='')
  
  setwd(ResultsDir)
  save(Results,file=name)
  #save performance out-of-sample
  name_m<-paste("Metrics_",Horizon,"SA_",Sys.Date(),".xlsx", sep='')
  
  Result<-do.call(rbind.data.frame,Metrics_City)
  setwd(ResultsDir)
  write.xlsx(Result,file=name_m)
  
}


#OSS forecasting based on best model
model              <-c("cubist","brnn","svmLinear2","brnn","cubist")
options            <-c(2,1,1,1,2) 
Models_Regions     <-list();
Preds_Regions      <-list()
h<-6
for(region in 1:length(City))
{
  option<-options[region]
  #Objects for region
  {
    preds_in           <-matrix(nrow=(length(data[[region]]$cum_confirmed)),ncol=2); #Recebe predições 
    colnames(preds_in) <-c("Preds","Errors");
    preds_OOS          <-as.data.frame(matrix(c(rep(0,h*3)),nrow=h,ncol=3))
    c                  <-Sys.Date()
    row.names(preds_OOS)<-as.Date(seq(as.Date(c)-2, by = "day", length.out = h))
    
    colnames(preds_OOS)<-c("Forecasting","LB","UB")
    Data_state         <-sort(c(rep(0,5),data[[region]]$cum_confirmed))
    #--------------------------Construindo Conjunto-----------------------------#
    data_m<-lags(Data_state, n = 5) #n representa o número de lags
    
    colnames(data_m)      <-c("Cum_Confirmed",paste("Lag",1:5,sep=""))
    
    fitControl2<- trainControl(method= "cv",number=5,savePredictions="final") 
    
    
    #Objects for Out-Of-Sample forecasting
    Y_train <-data_m[,1]
    X_train <-data_m[,-1]
    X_train_OOS<-matrix(c(rep(lags(tail(sort(data[[region]]$cum_confirmed)),n=5)[1:5],times=12)),nrow=12,ncol=5,byrow = TRUE)
    colnames(X_train_OOS)<-colnames(X_train)
    #----------------------Divisões para Treinamento----------------------------#
  }
  
  if(option==1)
  {
    set.seed(1234)
    Models_Regions[[region]]<-train(as.data.frame(X_train),as.vector(Y_train),
                                    method=model[region],
                                    preProcess = c("center","scale"), #Processamento Centro e Escala
                                    tuneLength= 4,                    #Número de tipos de parâmetros 
                                    trControl = fitControl2,verbose=FALSE)
    
    preds_in[,1]<-round(predict(Models_Regions[[region]],X_train))
    preds_in[,2]<-round(Y_train-preds_in[,1])
    
    #Preds OOS
    for(i in 1:h)
    {
      preds_OOS[i,1]<-round(predict(Models_Regions[[region]],as.data.frame(t(X_train_OOS[i,]))));
      preds_OOS[i,2]<-round(preds_OOS[i,1]-1.96*sd(preds_in[,2]));
      preds_OOS[i,3]<-round(preds_OOS[i,1]+1.96*sd(preds_in[,2]));
      
      #
      X_train_OOS[i+1,"Lag1"]<-preds_OOS[i,1]
      X_train_OOS[i+2,"Lag2"]<-preds_OOS[i,1]
      X_train_OOS[i+3,"Lag3"]<-preds_OOS[i,1]
      X_train_OOS[i+4,"Lag4"]<-preds_OOS[i,1]
      X_train_OOS[i+5,"Lag5"]<-preds_OOS[i,1]
      
    }
    
  }
  else if(option==2)
  {
    #Stacking ensemble
    modelst <- caretList(as.data.frame(X_train),as.vector(Y_train),
                         trControl=fitControl2, 
                         preProcess = c("center","scale"),
                         methodList=models)
    
    Models_Regions[[region]] <-caretStack(modelst, 
                                          trControl=fitControl2, 
                                          method=model[region],
                                          preProcess = c("center","scale"),
                                          tuneLength= 5)
    
    preds_in[,1]<-round(predict(Models_Regions[[region]],X_train))
    preds_in[,2]<-round(Y_train-preds_in[,1])
    
    #Preds OOS
    
    for(i in 1:h)
    {
      preds_OOS[i,1]<-predict(Models_Regions[[region]],as.data.frame(t(X_train_OOS[i,])));
      preds_OOS[i,2]<-preds_OOS[i,1]-1.96*sd(preds_in[,2]);
      preds_OOS[i,3]<-preds_OOS[i,1]+1.96*sd(preds_in[,2]);
      
      #
      X_train_OOS[i+1,"Lag1"]<-preds_OOS[i,1]
      X_train_OOS[i+2,"Lag2"]<-preds_OOS[i,1]
      X_train_OOS[i+3,"Lag3"]<-preds_OOS[i,1]
      X_train_OOS[i+4,"Lag4"]<-preds_OOS[i,1]
      X_train_OOS[i+5,"Lag5"]<-preds_OOS[i,1]
      
    }
  }
  else
  {
    
    Models_Regions[[region]]<-auto.arima(sort(Y_train))
    preds_arima             <-data.frame(forecast::forecast(Models_Regions[[region]], h=6))
    preds_OOS               <-as.vector(preds_arima[,c(1,4,5)])
  }
  
  Preds_Regions[[region]]<-preds_OOS
}
names(Preds_Regions)<-City
lapply(Preds_Regions,round)

################################Results-Extraction######################

setwd(ResultsDir)

load("Results_6SA_2020-08-17.RData");C6SA<-Results

models_names<-c("SVR","BRNN","KNN","BAGGING","Stacking","ARIMA")

for(i in 1:5)
{
  colnames(C6SA[[1]][[i]])<-models_names;  colnames(C6SA[[2]][[i]])<-models_names
  colnames(C6SA[[4]][[i]])<-models_names;  colnames(C6SA[[3]][[i]])<-models_names
  
}

#Best Models Names
{
  
  Names3SA<-c(sapply(C6SA[[8]],
                     function(x){rownames(x)[which.min(apply(x,MARGIN=1,min))]}))
  Names3SA<-c("Stacking", "Stacking","Stacking","Stacking","Stacking")
  
}

table(c(Names3SA))
#Models for each state
Best_Models<-list()
Conf_Interv<-list()
for(i in 1:5)
{
  aux<-matrix(nrow=dim(C6SA[[1]][[i]])[1]+6,ncol=2)
  SD<-sd(C6SA[[4]][[i]][,Names3SA[i]])
  
  
  aux[,1]<-c(C6SA[[1]][[i]][,Names3SA[i]],C6SA[[2]][[i]][,Names3SA[i]])

  aux[,2]<-data[[i]][1:dim(data[[i]])[1],"cum_confirmed"]
  
  aux1<-data.frame(Confirmed=c(aux[,1:2]),
                   Models=rep(c(#outer(c("ODA-"),Names1SA[i], FUN = "paste0"),
                                #outer(c("TDA-"),Names2SA[i],  FUN = "paste0"),
                                "Predicted",
                                "Observed"),each=dim(aux)[1]),
                   Date=rep(data[[i]][1:dim(data[[i]])[1],"date"],times=2))
    Best_Models[[i]]<-aux1
}
names(Best_Models)<-City
setwd(CodesDir)
source("Covid_Plot.R")
setwd(FiguresDir)

for(i in 1:5)
{
  nameps<-paste("PO_",City[i],".eps",sep="")
  
  x11()
  PO_Covid(Best_Models[[i]])
  
  ggsave(nameps, device=cairo_ps,width = 11,height = 9,dpi = 1200)
  
}