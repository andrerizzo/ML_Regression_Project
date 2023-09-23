# Load csv files to SQLite

# Load requires libraries
library(DBI)
library(RSQLite)
library(stringr)

# Create database in SQLite
db_connection = dbConnect(drv = SQLite(), dbname = './data/processed/used_cars.db')

# Get CSV files names
csv_files = list.files(path = "./data/raw", pattern = "csv")
size = length(csv_files)
for (aux in csv_files) {
  # Prepare string
  table_name = paste("data/raw/", aux, sep = "")
  
  # Load csv file to a dataframe 
  df = readr::read_csv(table_name)
  
  # Remove the suffix ".csv" from the table name
  table_name = str_extract(aux, regex("\\w*"))
  
  # Adjust table before write to DB
  df$year = as.integer(df$year)
  df$price = as.numeric(df$price)
  df$mileage = as.numeric(df$mileage) 
 
  # Write dataframe to a SQLite table
  dbWriteTable(conn = db_connection, name = table_name, value = df, overwrite = TRUE)
}

dbDisconnect(db_connection)

