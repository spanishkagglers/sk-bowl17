library(data.table)

source('../competition_config.R')
setwd(path_working_directory)

labels <- fread(path_stage1_labels)
setnames(labels,'id','ct_scan_id')


nodfiles <- list.files(path_10_features_nodules,pattern='*.csv',full.names=T)


compute<-function(ln,prefix){
  
  if(dim(ln)[1]>0){
    
    # Totales
    cols_total <- c("nod_cuboid_volume","nod_sphere_volume","nod_volume")
    dt_total <- ln[,lapply(.SD,sum,na.rm=T),.SDcols=cols_total]
    setnames(dt_total,cols_total,paste0('total_',cols_total))
    dt_total$total_nods = ln[,.N]
    

    cols_total_tissue <- c("nod_blood_ratio","nod_bone_ratio","nod_csf_ratio","nod_fat_ratio",
                           "nod_greymatter_ratio","nod_kidney_ratio","nod_liver_ratio","nod_muscle_ratio",
                           "nod_soft_ratio","nod_water_ratio","nod_whitematter_ratio")
    for(f in cols_total_tissue){
      varTotal <- paste0('total_',strsplit(f,'_')[[1]][2:2])
      dt_total[[varTotal]] <- ln[,sum(get(f)*nod_volume)]
      varTotalRatio <- paste0(varTotal,'_ratio')
      dt_total[[varTotalRatio]] <- dt_total[,get(varTotal)/total_nod_volume]
    }
    
    
    
    
    cols <- names(ln)[!names(ln) %in% c('ct_scan_id','id')]
    
    dt_mean <- ln[,lapply(.SD,mean,na.rm=T),.SDcols=cols]
    setnames(dt_mean,cols,paste0('mean_',cols))
    
    dt_sd <- ln[,lapply(.SD,sd,na.rm=T),.SDcols=cols]
    setnames(dt_sd,cols,paste0('sd_',cols))
    
    dt_max <- ln[,lapply(.SD,max,na.rm=T),.SDcols=cols]
    setnames(dt_max,cols,paste0('max_',cols))
    
    dt_min <- ln[,lapply(.SD,min,na.rm=T),.SDcols=cols]
    setnames(dt_min,cols,paste0('min_',cols))
    
    dt_difMaxMin <- dt_max - dt_min
    setnames(dt_difMaxMin,names(dt_difMaxMin),paste0('difMaxMin_',cols))
    
    dt_ratioMaxMin <- dt_max / dt_min
    setnames(dt_ratioMaxMin,names(dt_ratioMaxMin),paste0('ratioMaxMin_',cols))
    for(f in names(dt_ratioMaxMin)){
      dt_ratioMaxMin[is.na(get(f)),(f):=NA]
      dt_ratioMaxMin[is.infinite(get(f)),(f):=NA]
    }
    
    dt <- cbind(dt_total,dt_mean,dt_sd,dt_max,dt_min,dt_difMaxMin,dt_ratioMaxMin)
    setnames(dt,names(dt),paste0(prefix,names(dt)))
    
  }else{
    dt <- NA
  }
  
  
  return(dt)
}

# Incluyo CNN intermediate
fileName <- list.files(path_intermediate_cnn,pattern='intermediate_output_',full.names=T)
cnn <- fread(fileName, showProgress = TRUE, header=T)
cnnNames <- names(cnn)[!names(cnn)%in%c('nodule_id')]
setnames(cnn,cnnNames,paste0('cnn_',cnnNames))

# Incluyo predicciones a nivel nodulos yTrainOOFpreds y yTestPreds
yTrainOOFfileName = 'yTrainPredsOOF_xgb4_all_cv5_svm0.5661783_stm0.5232184_esr100_ss0.7_md5_csbt0.9_eta0.05_mcw20_d2017.03.28_h16.40.58.csv'
yTestfileName = 'yTestPreds_xgb4_all_cv5_svm0.5661783_stm0.5232184_esr100_ss0.7_md5_csbt0.9_eta0.05_mcw20_d2017.03.28_h16.40.58.csv'
yTrainOOF <- fread(paste0(path_xgb_nodule_preds,yTrainOOFfileName), showProgress = TRUE, header=T)
yTest <- fread(paste0(path_xgb_nodule_preds,yTestfileName), showProgress = TRUE, header=T)
yNodPreds <- rbindlist(list(yTrainOOF,yTest))
yNodPredsNames <- names(yNodPreds)[!names(yNodPreds)%in%c('id')]
setnames(yNodPreds,yNodPredsNames,'xgbNodPred')

allData = data.table()

contador = 1
for(nodfile in nodfiles){
  
  print(paste0(contador, " : ", nodfile))
  
  #nodfile = nodfiles[1]
  
  ln <- fread(nodfile)
  
  #ln <- ln[nod_sphere_radius>12 & nod_sphere_radius<120,]
  
  ######################################## voy por aquí ##############################################
  setkey(ln,id)
  setkey(yNodPreds,id)
  ln <- ln[yNodPreds,nomatch=0]
  ######################################## voy por aquí ##############################################
  
  if(dim(ln)[1]){
    setnames(ln,'id','nodule_id')
    setkey(ln,nodule_id)
    setkey(cnn,nodule_id)
    ln <- ln[cnn,nomatch=0]
    ln[,nodule_id:=NULL]
    
    ln[,nod_geo_to_massCenter_distance:=sqrt((raw_nod_center_x-raw_nod_masscenter_x)^2 + (raw_nod_center_y-raw_nod_masscenter_y)^2 + (raw_nod_center_z-raw_nod_masscenter_z)^2)]
    
    selected_cols = names(ln)[!grepl('raw_cuboid_',names(ln)) & !grepl('raw_roi_',names(ln))]
    ln <- ln[,.SD,.SDcols=selected_cols]
    
    id_scan <- unique(ln$ct_scan_id)
    
    label <- ifelse(labels[ct_scan_id==id_scan,.N]==0,NA,labels[ct_scan_id==id_scan,cancer])
    
    # squared features
    selected_cols = names(ln)[!names(ln)%in%c('ct_scan_id')]
    for(feat in selected_cols){
      featName = paste0('squared_',feat)
      ln[,(featName):=(get(feat))^2]
    }
    
    allData <- rbindlist(list(
      allData,
      data.table(
        id=id_scan,
        compute(ln,'allNods_'),
        cancer=label
      )
    ),fill=TRUE,use.names=TRUE)
  }
  contador = contador + 1
}

# Elimino features constantes
cols <- names(allData)[!names(allData) %in% c('id','cancer')]
for(f in cols){
  if(max(allData[[f]],na.rm=T)==min(allData[[f]],na.rm=T)){
    allData[,(f):=NULL]
  }
}


train <- allData[!is.na(cancer),]
test  <- allData[is.na(cancer),]

fileName <- paste0(path_13_features_lungs,'lung_features.csv')
fwrite(x=allData,file=fileName,row.names=F,quote=TRUE)

fileName <- paste0(path_ml_input,'train_lung.csv')
fwrite(x=train,file=fileName,row.names=F,quote=TRUE)

fileName <- paste0(path_ml_input,'test_lung.csv')
fwrite(x=test,file=fileName,row.names=F,quote=TRUE)
