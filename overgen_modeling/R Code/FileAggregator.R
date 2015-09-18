# This reads in existing LMP data files from a directory, and merges them into a unit for casting.
# Must previously run a separate script to pull out all of the .csv files that need to be processed
# Only uses the LMP data, not MCC, MCL, or MCE.

rm(list=ls(all= TRUE))
setwd("~/Google Drive/CE 290 Project/Data Collection/Prices/R code")
library(reshape2)
source('CAISO-oasisAPI-operations_em.r')

start_date <- 20140101
end_date <- 20140531
saveFileName <- "All_PNodes_MCE_Aggregated_JanMay_2014"

activeDay <- strptime(start_date,"%Y%m%d")
endDay <- strptime(end_date,"%Y%m%d")

data <- data.frame()

while (activeDay <= endDay){
  # Pulls in each new day and binds it to the dataframe "data"
  myClock <- proc.time()
  
  # File names are of the format 20130101_20130101_PRC_LMP_DAM_LMP.csv
  myFileName <- paste("./data/",strftime(activeDay,"%Y%m%d"),"_",strftime(activeDay,"%Y%m%d"),"_PRC_LMP_DAM_MCE.csv",sep="")
  newData <- read.csv(myFileName)
  data <- rbind(data,newData)
  
  myTimeDiff <- proc.time() - myClock
  print(paste("Processed date", strftime(activeDay,"%Y%m%d"),"in ",toString(myTimeDiff[1])))
  activeDay <- activeDay + 86400
}

print("Done with loop; melting data")
myClock <- proc.time()
meltedData <- meltCAISO(data)
remove("data")
myTimeDiff <- proc.time() - myClock

print(paste("Finished melting data in ", toString(myTimeDiff[1])," s; filtering for desired nodes"))

save.image("MyMeltedData.RData")

myClock <- proc.time()
# The file desiredNodes contains a list of nodes whose data is actually desired
# The names of the LMP Nodes for which we have both OASIS prices and geographic lat/long data are included in the list below
#desiredNodes <- read.csv("~/Google Drive/CE 290 Project/Data Collection/Prices/LMP_NodeList_ToUse.csv",header=F,as.is=TRUE)
# This filters the full set of data by the desiredNodes
#meltedData <- meltedData[meltedData$NODE %in% desiredNodes[,1] ,]

myTimeDiff <- proc.time() - myClock

# Cast into a usable form
print(paste("Finished filtering data in ", toString(myTimeDiff[1])," s, Casting..."))

myClock <- proc.time()
myCasting<-acast(meltedData,OPR_DT+variable~NODE) # |LMP_TYPE
#myCasting<-acast(meltedData,OPR_DT+variable~NODE) # |LMP_TYPE


myTimeDiff <- proc.time() - myClock
print(paste("Finished Casting data in ", toString(myTimeDiff[1])," s, Writing to file..."))


myfile <- file(paste(saveFileName,"short.csv",sep=""))


write.csv(t(myCasting),myfile)

