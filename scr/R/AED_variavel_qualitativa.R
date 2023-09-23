# Script para a realização de AED em variáveis qualitativas nominais e ordinais
# Autor: André Rizzo
# Data: 08/05/2023
# Verão: 2.0


tabela_de_frequencias_var_qualitativa = function(dataframe_variavel) {
  
  # Carrega bibliotecas necessárias
  library(summarytools)
  
  # Cria tabela de frequências
  summarytools::freq(x = dataframe_variavel, order = "freq")

}


analise_valores_faltantes = function(dataframe_variavel){
  cat("ESTUDO DE VALORES FALTANTES \n")
  cat("Número de NAs:", sum(is.na(dataframe_variavel)),"\n")
  cat("Percentual de NAs:", sum(is.na(dataframe_variavel))/length(dataframe_variavel),"% \n \n")
}
  
  

analise_cardinalidade = function(dataframe, dataframe_variavel){
  
  cat("ESTUDO DA CARDINALIDADE \n")
  total_observacoes = length(dataframe_variavel)
  cat("Total de observações: ", total_observacoes[1], "\n")
  observacoes_unicas = length(unique(dataframe_variavel))
  cat("Total de observações únicas: ", observacoes_unicas, "\n")
  proporcao_obs_unicas = (observacoes_unicas / total_observacoes)*100
  cat("% de observações únicas: ", proporcao_obs_unicas,"% \n \n")
}


grafico_de_barras_var_qualitativa = function(dataframe, variavel, dataframe_variavel) {
  
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
#analise_cardinalidade(dataframe = df, dataframe_variavel = df$model)
#grafico_de_barras(df_audi, "year")


