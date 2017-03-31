library(data.table)
library(Metrics)
library(caret)
library(xgboost)

source('../competition_config.R')
setwd(path_working_directory)


trainName = 'train_lung_nods.csv'
testName = 'test_lung_nods.csv'

# load data
import.cols = 'all' # 'all', 'selected'

# split
n_folds = 5                    

# xgb params
alg='xgb4'
early_stopping = 100
print_every    = 10

xgb.params <- list(booster = 'gbtree'
                   , objective = 'binary:logistic'
                   , subsample = c(0.6,0.7,0.8)
                   , max_depth = c(3,5,7)
                   , colsample_bytree = c(0.5,0.7,0.9)
                   , eta = c(0.01,0.03,0.05)
                   , min_child_weight = c(1,2,3,4,10,20)
)
  

# save to log
log <- c()
log <- c(log,'')
log <- c(log,paste0("import.cols:", import.cols))
log <- c(log,paste0("n_folds:", n_folds))
log <- c(log,paste0("early_stopping:", early_stopping))
log <- c(log,'')

# import

if(import.cols=='all'){
  TRAIN <- fread(paste0(path_ml_input,trainName), showProgress = TRUE, header=T)
  test <- fread(paste0(path_ml_input,testName),  showProgress = TRUE, header=T)
  log <- c(log,'Import: all features')
}else if(import.cols=='selected'){
  colsToImport = c(
    'id',
    'allNods_sd_nod_center_to_roi_center_distance',
    'allNods_min_nod_cuboid_most_asimetric_face_edges_ratio',
    'allNods_ratioMaxMin_nod_center_to_roi_center_distance',
    'allNods_difMaxMin_nod_ymax_ymaxRoi_ratio',
    'allNods_sd_nod_hu_mean_vs_cuboid_hu_mean',
    'allNods_sd_nod_bone_ratio',
    'allNods_difMaxMin_nod_center_to_xaxis_angle',
    'allNods_ratioMaxMin_nod_zcenter_zcenterRoi_ratio',
    'allNods_ratioMaxMin_nod_cuboid_area_volume_ratio',
    'allNods_sd_nod_xmax_xmaxRoi_ratio',
    'allNods_ratioMaxMin_nod_center_to_xaxis_angle',
    'allNods_difMaxMin_nod_zmax_zmaxRoi_ratio',
    'allNods_mean_nod_cuboid_area_volume_ratio',
    'allNods_min_nod_center_to_zaxis_angle',
    'allNods_difMaxMin_nod_cuboid_area_volume_ratio',
    'allNods_min_nod_center_to_roi_center_distance',
    'allNods_sd_nod_center_to_yaxis_angle',
    'allNods_difMaxMin_nod_zcenter_zcenterRoi_ratio',
    'allNods_difMaxMin_nod_center_to_yaxis_angle',
    'allNods_min_nod_cuboid_filled_ratio',
    'allNods_min_nod_center_to_xaxis_angle',
    'allNods_sd_nod_center_to_zaxis_angle',
    'allNods_min_nod_center_to_yaxis_angle',
    'allNods_min_nod_sphere_filled_ratio',
    'allNods_sd_nod_ycenter_ycenterRoi_ratio',
    'allNods_mean_nod_cuboid_diagonal',
    'allNods_max_nod_center_to_xaxis_angle',
    'allNods_max_nod_hu_mean',
    'allNods_ratioMaxMin_nod_center_to_zaxis_angle',
    'allNods_sd_nod_hu_mode',
    'allNods_ratioMaxMin_nod_sphere_radius',
    'allNods_mean_nod_center_to_yaxis_angle',
    'allNods_min_nod_xmax_xmaxRoi_ratio',
    'allNods_max_nod_hu_mode',
    'allNods_mean_nod_cuboid_filled_ratio',
    'allNods_ratioMaxMin_nod_xmax_xmaxRoi_ratio',
    'allNods_ratioMaxMin_nod_cuboid_volume',
    'cancer')
  TRAIN <- fread(paste0(path_ml_input,trainName), showProgress = TRUE, header=T, select=colsToImport)
  test <- fread(paste0(path_ml_input,testName),  showProgress = TRUE, header=T, select=colsToImport)
  log <- c(log,'Import: selected features',paste0(colsToImport,collapse=", "))
}
log <- c(log,'')


# set target name
setnames(TRAIN,'cancer','target')
setnames(test,'cancer','target')

# order
setorder(TRAIN,id)
setorder(test,id)

# merge
allData <- rbindlist(list(TRAIN, test), use.names=T,fill=T)
remove(TRAIN,test)
gc()


# Clase de cada feature
allData.x.clases = data.table(feature=names(allData), clase=allData[,.(cl=lapply(.SD,class))]$cl)
allData.x.clases = allData.x.clases[!feature %in% c('id','target')]


# Normalizo numericas
feats.numericas = allData.x.clases[clase %in% c('numeric','integer'),feature]
for (f in colnames(allData)[colnames(allData) %in% feats.numericas]) {
  allData[,(f):=scale(get(f))]
}


# allData to TRAIN and test
TRAIN <- allData[!is.na(target),]
test  <- allData[is.na(target),]
remove(allData)
gc()


# split data
set.seed(1)
nrows.TRAIN = length(TRAIN$target)
foldsVal <- createFolds(factor(TRAIN$target), k = n_folds)
foldsTrain = list()
for(j in 1:n_folds){
  foldsTrain[j][[1]] = c(1:nrows.TRAIN)[!c(1:nrows.TRAIN) %in% foldsVal[j][[1]]]
}


log <- c(log,'')
log_prepro = copy(log)


# create combinations y shuffle
set.seed(2017)
combs = data.table(expand.grid(xgb.params,stringsAsFactors = F))
combs = combs[sample(nrow(combs)),]
print(paste0('Numero combinaciones:',nrow(combs)))


for(comb in 1:nrow(combs)){
  
  # Ejecuto esta combinacion, salvo que ya la haya ejecutado
  ejecutar_combinacion = T
  
  comb_actual = c(early_stopping,as.character(combs[comb,3:7]))
  
  # Leo combinaciones ya generadas
  comb_generadas = list.files(path_ml_output,pattern = 'yTestPreds_xgb4')
  comb_generadas = strsplit(comb_generadas,'_')
  if(length(comb_generadas)){
    for(k in 1:length(comb_generadas)){
      comb_generadas[[k]] = gsub('[a-z]','',comb_generadas[[k]][7:12])
      if(all(comb_generadas[[k]]==comb_actual)){
        ejecutar_combinacion = F
      }
    }
  }
  
  
  if(ejecutar_combinacion){
    
    print(paste0('Combinacion:',comb))
    print(combs[comb])
    
    timeInicial = Sys.time()
    timestmp = gsub('[:]','.',as.character(timeInicial))
    timestmp = gsub('[-]','.',timestmp)
    timestmp_day=strsplit(timestmp,' ')[[1]][1]
    timestmp_hour=strsplit(timestmp,' ')[[1]][2]
    timestmp = gsub('[ ]','_',timestmp)
  
    
    # save to log
    log = c(paste0("InicioTime:", timeInicial),log_prepro)
    log <- c(log,toString(names(combs[comb])))
    log <- c(log,toString(combs[comb]))
    log <- c(log,'')
    
    
    list.modelFoldN       <- list()
    list.fiFoldN          <- list()
    list.yValPredFoldN    <- list()
    list.yTestPredFoldN   <- list()
    list.scoreTrain       <- list()
    list.scoreVal         <- list()
    list.idsyValPredFoldN <- list()
    
    
    for (i in 1:n_folds) {
      
      # train and val
      train = TRAIN[foldsTrain[i][[1]], ]
      val   = TRAIN[foldsVal[i][[1]], ]
      testN = test
      
      # Convert from data.table to matrix
      xcols <- names(train)[!names(train) %in% c("id", "target")]
      train.x_m <- as.matrix(model.matrix(object=~ ., data=model.frame(formula=~ ., data=train[,.SD,.SDcols=xcols], na.action="na.pass")))
      val.x_m   <- as.matrix(model.matrix(object=~ ., data=model.frame(formula=~ ., data=  val[,.SD,.SDcols=xcols], na.action="na.pass")))
      test.x_m  <- as.matrix(model.matrix(object=~ ., data=model.frame(formula=~ ., data= testN[,.SD,.SDcols=xcols], na.action="na.pass")))
      
      # Convert from matrix to dMatrix
      train.xy_dm <- xgb.DMatrix(train.x_m, label=train$target, missing=NA)
      val.xy_dm   <- xgb.DMatrix(val.x_m, label=val$target, missing=NA)
      test.x_dm   <- xgb.DMatrix(test.x_m, missing=NA)
      
      # Model training xgb
      print('')
      print(paste0("Fold ",i,". Training xgboost starting time : ",Sys.time()))
      set.seed(1)
      model <- xgb.train(params = as.list(combs[comb])
                       , data = train.xy_dm
                       , nrounds = 10000
                       , verbose = 1
                       , print.every.n = print_every
                       , eval_metric = 'logloss'
                       , watchlist = list(eval = val.xy_dm, train = train.xy_dm)
                       , early.stop.round = early_stopping
                       , maximize = FALSE
                       )
      
      # Feature importance xgb
      fi = xgb.importance(colnames(train.x_m), model=model)
      
      # Predict xgb
      yValPreds   <- predict(model, val.xy_dm, ntreelimit=model$bestInd)
      yTrainPreds <- predict(model, train.xy_dm, ntreelimit=model$bestInd)
      yTestPreds  <- predict(model, test.x_dm, ntreelimit=model$bestInd)
      
      # Scores
      scoreVal   = round(Metrics::logLoss(val$target, yValPreds),7)
      scoreTrain = round(Metrics::logLoss(train$target, yTrainPreds),7)
      
      # Print
      log.txt = paste("Fold:",i," numRound:",model$bestInd," ScoreVal:",scoreVal," ScoreTrain:",scoreTrain," Time:", Sys.time())
      log <- c(log, log.txt)
      
      # Save each fold results
      list.modelFoldN[[i]]       <- model
      list.fiFoldN[[i]]          <- fi
      list.yValPredFoldN[[i]]    <- yValPreds
      list.yTestPredFoldN[[i]]   <- yTestPreds
      list.scoreTrain[[i]]       <- scoreTrain
      list.scoreVal[[i]]         <- scoreVal
      list.idsyValPredFoldN[[i]] <- val$id
      
    }
    
    log <- c(log,'')
    rm(train,val,testN)
    gc()
    
    # yTestPredsCV
    yTestPredsCV <- as.data.table(list.yTestPredFoldN)
    yTestPredsCV[, meanTarget := rowMeans(.SD)]
    yTestPredsCV$id <- test$id
    
    # yValPredsCV, fi
    fiFoldN <- data.table()
    yValPredsCV <- data.table()
    for (i in 1:n_folds) {
      yValPredsCV = rbindlist(list(yValPredsCV, data.table(id=list.idsyValPredFoldN[[i]], pred=list.yValPredFoldN[[i]]) ))
      fiFoldN = rbindlist(list(fiFoldN, cbind(data.table(list.fiFoldN[[1]]),Fold=i)),use.names=T)
    }
    fiFoldN = rbindlist(list(
      fiFoldN,
      fiFoldN[,.(Gain=mean(Gain),Cover=mean(Cover),Frequence=mean(Frequence),Fold=0),by=.(Feature)]
    ))[order(Fold,-Gain)]
      
    
    # Ordeno yValPredsCV por id, segun el orden de los ids de TRAIN$id
    yValPredsCV = yValPredsCV[match(TRAIN$id, id),]
    yValPredsCV = yValPredsCV[!is.na(pred),] 
    
    
    # Guardo los ids presentes en yValPreds
    idsyValPredsCV = yValPredsCV$id
    
    
    # ScoresCV
    scoreValMeanCV   = round(mean(unlist(list.scoreVal)),7)
    scoreTrainMeanCV = round(mean(unlist(list.scoreTrain)),7)
    
    log <- c(log,paste("FinTime:", Sys.time()))
    
    fileNameComun = paste0(alg,'_',import.cols,'_cv',n_folds,'_svm',scoreValMeanCV,'_stm',scoreTrainMeanCV,
                           '_esr',early_stopping,'_ss',combs[comb]$subsample,'_md',combs[comb]$max_depth,'_csbt',combs[comb]$colsample_bytree,'_eta',combs[comb]$eta,'_mcw',combs[comb]$min_child_weight,
                           '_d',timestmp_day,'_h',timestmp_hour,'.csv')
  
    fwrite(x=list(log), file=paste0(path_ml_output,'log_',fileNameComun), row.names=F, quote=FALSE)
    fwrite(x=fiFoldN,  file=paste0(path_ml_output,'fiFolds_',fileNameComun), row.names=F, quote=FALSE)
    fwrite(x=yTestPredsCV[,.(id,cancer=meanTarget)], file=paste0(path_ml_output,'yTestPreds_',fileNameComun), row.names=F, quote=FALSE)
    fwrite(x=yValPredsCV,file=paste0(path_ml_output,'yTrainPredsOOF_',fileNameComun), row.names=F, quote=FALSE)
    
    print(paste0("Combinacion ",comb," finalizada a las ",Sys.time()))
    
  }else{
    print(paste0("Combinacion ",comb,' no ejecutada, pues ya existia esta combinacion'))
    print(combs[comb])
  }

}