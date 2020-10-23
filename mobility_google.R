library(dplyr)

download_mobility_table <- function(url){
  tmp <- tempfile()
  download.file(url, tmp)
  response <- read.csv(tmp, encoding = "UTF-8")
  unlink(tmp)
  return(response)
}

url <- 'https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv?cachebust=dfad08836e5fd8d1'

mobility <- download_mobility_table(url)

mobility.br <- mobility %>% filter(country_region_code == "BR")