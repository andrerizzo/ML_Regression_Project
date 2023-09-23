# Script para a realização de análise bivariada
# Autor: André Rizzo
# Data: 21/08/2023
# Verão: 1.0




###########################################################################
# Análise de variáveis categóricas (independetes) e numéricas (dependentes)
###########################################################################

boxplot_categorica_numerica = function(dataframe, var_independente, var_dependente) {
  library(ggplot2)
  library(stringr)
  
  ggplot(dataframe, aes(x=.data[[var_independente]], y=.data[[var_dependente]])) +
    geom_boxplot(fill = "green") +
    labs(title = paste0("Boxplot das Variaveis ", str_to_title(var_independente), 
                        " e  ", str_to_title(var_dependente)))
}


# Teste de independência variáveis categóricas (independetes) e numéricas (dependentes)

# Kendall Rank Correlation (postos)
# Ho:
# H1:
# p-value < 0.05 -> Reject Ho
categorica_numerica_correlacao = function(df_var_independente, df_var_dependente) {
  x_rank = rank(df_var_independente)
  test = cor.test(x = x_rank, y = df_var_dependente, method = "kendall")
  print(test)
  if(test$p.value <= 0.05) {
    cat("As variáveis são correlacionadas")
  } else {
    cat("As variáveis NÃO SÃO correlacionadas")
  }
}




##############################################################################
# Análise de variáveis categóricas (independetes) e categóricas (dependentes)
##############################################################################

categorica_categorica_correlacao = function(df_var_independente, df_var_dependente) {
  test = chisq.test(x = df_var_independente, y = df_var_dependente)
  print(test)
  if(test$p.value <= 0.05) {
    cat("As variáveis são correlacionadas")
  } else {
    cat("As variáveis NÃO SÃO correlacionadas")
  }
}




