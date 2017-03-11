library(data.table)

source('../competition_config.R')
setwd(path_working_directory)

labels <- fread(path_stage1_labels)
setnames(labels,'id','ct_scan_id')


nodfiles <- list.files(path_10_features_nodules,pattern='*.csv',full.names=T)


compute<-function(ln,prefix){
  
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
  
  dt <- cbind(dt_mean,dt_sd,dt_max,dt_min,dt_difMaxMin,dt_ratioMaxMin)
  setnames(dt,names(dt),paste0(prefix,names(dt)))
  
  return(dt)
}


allData = data.table()

for(nodfile in nodfiles){
  
  ln <- fread(nodfile)
  id_scan <- unique(ln$ct_scan_id)

  ###############################
  # Inserto features en allData
  ###############################
  
  label <- ifelse(labels[ct_scan_id==id_scan,.N]==0,NA,labels[ct_scan_id==id_scan,cancer])

  allData <- rbindlist(list(
    allData,
    data.table(
      id=id_scan,
      compute(ln,'allNods_'),
      cancer=label
    )
  ))

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

fileName <- paste0(path_ml_input,'train.csv')
fwrite(x=train,file=fileName,row.names=F,quote=TRUE)

fileName <- paste0(path_ml_input,'test.csv')
fwrite(x=test,file=fileName,row.names=F,quote=TRUE)
