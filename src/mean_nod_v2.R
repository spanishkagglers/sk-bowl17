library(data.table)


fname='C:\\Proyectos\\ML\\dsb\\sk-bowl17\\output\\ML\\output\\XGB_Nodule_Predictions\\yTestPreds_xgb4_all_cv5_svm0.5655347_stm0.5097876_esr100_ss0.8_md7_csbt0.7_eta0.03_mcw20_d2017.03.29_h21.53.11.csv'
dt=fread(fname)
dt[,id:=strsplit(id,'_')[[1]][1],by=id]



dt=dt[,.(cancer=mean(cancer,na.rm=T)),by=id]
fwrite(x=dt,file='C:\\Proyectos\\ML\\dsb\\sk-bowl17\\output\\ML\\output\\sub_20170329_mean_lung_with_cnn512.csv')
