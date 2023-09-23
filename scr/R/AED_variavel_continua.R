# Script para a realização de análise univariada
# Autor: André Rizzo
# Data: 11/05/2023
# Verão: 2.0


analise_valores_faltantes_var_continua = function(dataframe_variavel){
  cat("\n")
  cat("ESTUDO DE VALORES FALTANTES \n")
  cat("Número de NAs:", sum(is.na(dataframe_variavel)),"\n")
  cat("Percentual de NAs:", sum(is.na(dataframe_variavel))/length(dataframe_variavel),"% \n \n")
}


analise_numerica_var_continua = function(dataframe_variavel) {
  
  cat("ESTUDO DAS MEDIDAS DE POSIÇÃO\n")
  cat("Média = ", mean(dataframe_variavel ), "\n")
  cat("Mediana = ", median(dataframe_variavel ), "\n")
  cat("Moda = ", max(dataframe_variavel ), "\n")
  cat("Mínimo = ", min(dataframe_variavel ), "\n")
  cat("Máximo = ", max(dataframe_variavel ), "\n")
  cat("\n")
  cat("ESTUDO DAS MEDIDAS DE DISPERSÃO\n")
  cat("Amplitude = ", max(dataframe_variavel ) - min(dataframe_variavel ), "\n")
  cat("Variância = ", var(dataframe_variavel ),"\n")
  cat("Desvio padrão = ", sd(dataframe_variavel ), "\n")
  cat("IQR = ", IQR(dataframe_variavel ),"\n")
  cat("\n")
  cat("ESTUDO DAS MEDIDAS DE ASSIMETRIA\n")
  assimetria = skewness(dataframe_variavel )
  cat("Coeficiente de assimetria = ", assimetria, "\n")
  if (assimetria >= -0.5 & assimetria <= 0.5) {
    diag_assimetria_1 = "Distribuição simétrica"
    cat("Grau de assimetria: ", diag_assimetria_1, "\n")
  } else if (assimetria < -1 | assimetria > 1) {
    diag_assimetria_1 = "Alto"
    cat("Grau de assimetria: ", diag_assimetria_1, "\n")
  } else if ((assimetria >= -1 & assimetria < 0.5) | (assimetria > 0.5 & assimetria <= 1)) {
    diag_assimetria_1 = "Médio"
    cat("Grau de assimetria: ", diag_assimetria_1, "\n")
  }
  
  
  if (assimetria >= -1 & assimetria <= 1) {
    diag_assimetria_2 = "Distribuição simétrica."
    cat("Tipo de assimetria: ", diag_assimetria_2, "\n")
  } else if (assimetria > 1) {
    diag_assimetria_2 = "Positiva (direita)"
    cat("Tipo de assimetria: ", diag_assimetria_2, "\n")
  } else if (assimetria < -1) {
    diag_assimetria_2 = "Negativa (esquerda)."
    cat("Tipo de assimetria: ", diag_assimetria_2, "\n")
  }
  
  cat("\n")
  cat("ESTUDO DO FORMATO DA CURVA\n")
  curtose = kurtosis(dataframe_variavel)
  cat("Coeficiente de curtose = ", curtose, "\n")
  if (curtose == 3) {
    diag_curtose = "Distribuição mesocúrtica (semelhante à Normal)"
    cat("Análise: ", diag_curtose, "\n")
  } else if (curtose < 0) {
    diag_curtose = "Distribuição platicúrtica (achatada)."
    cat("Análise: ", diag_curtose, "\n")
  } else if (curtose > 3) {
    diag_curtose = "Distribuição leptocúrtica (afunilada)"
    cat("Análise: ", diag_curtose, "\n")
  }
  
}


histograma = function(dataframe, variavel) {
  library(ggplot2)
  library(stringr)
  library(scales)
  
  ggplot(dataframe, aes(x=.data[[variavel]])) +
    geom_histogram(aes(y = stat(count)), fill = "cornflowerblue", color = "black", alpha = 0.5) +
    labs(title = paste0("Histograma da Variavel ", str_to_title(variavel)))
    #geom_density(alpha = 0.4, colour = "red") +
    #scale_y_continuous(labels = label_comma())
}


density_plot = function(dataframe, variavel) {
  library(ggplot2)
  library(stringr)
  #library(scales)
  
  ggplot(dataframe, aes(x=.data[[variavel]])) +
    geom_density(aes(y = stat(count)), color = "red", lwd = 0.8) +
    labs(title = paste0("Density Plot da Variavel ", str_to_title(variavel)), y = "Frequencia")
  
}


box_plot = function(dataframe, variavel) {
  library(ggplot2)
  library(stringr)
  
  ggplot(dataframe, aes(y = .data[[variavel]])) +
    geom_boxplot(fill = "green", width = 2) +
    labs(title = paste0("Boxplot da Variavel ", str_to_title(variavel)), x = "")+
    theme_minimal()
}



#box_plot(dataframe = df, variavel = "price")
#density_plot(dataframe = df, variavel = "price")
#histograma(dataframe = df, variavel = "year")
#teste_normalidade_shapiro(variavel = df_audi$year)
