# Script para a realização de testes de normalidade
# Autor: André Rizzo
# Data: 12/05/2023
# Verão: 1.0


teste_normalidade_shapiro = function(dataframe_variavel, variavel) {
  library(nortest)
  
  amostra = sample(dataframe_variavel, 5000, replace = TRUE)
  normalidade_shap = shapiro.test(amostra)
  cat("\n")
  cat("\n")
  cat("Análise da Variável", toupper(variavel), "\n")
  print(normalidade_shap)
  
  if (normalidade_shap$p.value <= 0.05) {
    diag_normalidade_shap = "As observações não são normalmente distribuídas."
    cat("Interpretação: ", diag_normalidade_shap, "\n")
  } else if (normalidade_shap$p.value > 0.05){
    diag_normalidade_shap = "As observações são normalmente distribuídas."
    cat("Interpretação: ", diag_normalidade_shap, "\n")
  } 
  
}


teste_normalidade_anderson_darling = function(dataframe_variavel) {
  library(nortest)
  
  normalidade_ad = ad.test(dataframe_variavel)
  print(normalidade_ad)
  if (normalidade_ad$p.value <= 0.05) {
    diag_normalidade_ad = "As observações não são normalmente distribuídas."
    cat("Interpretação: ", diag_normalidade_ad, "\n")
  } else if (normalidade_ad$p.value > 0.05){
    diag_normalidade_ad = "As observações são normalmente distribuídas."
    cat("Interpretação: ", diag_normalidade_ad, "\n")
  } 
}



qqplot = function(dataframe, variavel) {
  library(stringr)
  
  ggplot(dataframe, aes(sample=.data[[variavel]]))+
    stat_qq(color = "green", size = 2.5) +
    stat_qq_line() +
    labs(title = paste0("Q-Q Plot da Variavel ",str_to_title(variavel)), y = variavel, x = "Distribuição Teórica")
}



#teste_normalidade_anderson_darling(dataframe_variavel = df$price)


#teste_normalidade_shapiro(dataframe_variavel = df$price, variavel = "price")
