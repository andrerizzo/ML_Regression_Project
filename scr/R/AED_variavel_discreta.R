# Script para a realização de AED em variáveis numéricas discretas
# Autor: André Rizzo
# Data: 13/08/2023
# Verão: 2.0


analise_valores_faltantes_var_discreta = function(dataframe_variavel){
  cat("ESTUDO DE VALORES FALTANTES \n")
  cat("Número de NAs:", sum(is.na(dataframe_variavel)),"\n")
  cat("Percentual de NAs:", sum(is.na(dataframe_variavel))/length(dataframe_variavel),"% \n \n")
  cat("\n")
}


analise_numerica_var_discreta = function(dataframe_variavel) {
  moda <- function(x) {
    u <- unique(dataframe_variavel)
    tab <- tabulate(match(x, u))
    u[tab == max(tab)]
  }
  
  cat("ESTUDO DAS MEDIDAS RESUMO \n")
  cat("Moda = ", moda(dataframe_variavel), "\n")
  cat("Mínimo = ", min(dataframe_variavel), "\n")
  cat("Máximo = ", max(dataframe_variavel), "\n")
  cat("\n \n")
}

tabela_de_frequencias_var_discreta = function(dataframe_variavel) {
  
  # Carrega bibliotecas necessárias
  library(summarytools)
  
  # Cria tabela de frequências
  summarytools::freq(x = dataframe_variavel, order = "freq")

}


grafico_de_barras_var_discreta = function(dataframe, variavel, dataframe_variavel) {
  
  # Carrega bibliotecas necessárias
  library(ggplot2)
  library(stringr)
  library(dplyr)
  
  
  ggplot(dataframe, aes(.data[[variavel]])) +
    geom_bar(width=0.8,fill="#4682B4") +
    labs(title = paste0("Barplot da variavel ", str_to_title(variavel)), y = "Frequencia")
  
  # Colocar label em todas as barras no eixo X
}



#tabela_de_frequencias(df_audi$year)
#grafico_de_barras(df_audi, "year")


