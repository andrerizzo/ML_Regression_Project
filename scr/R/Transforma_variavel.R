# Script para transformação de variáveis
# Autor: André Rizzo
# Data: 14/05/2023
# Verão: 1.0


transforma_variavel = function(dataframe, variavel, usar_orderNorm, usar_exponencial, out_sample) {

  #define packages to install
  packages <- c("nortest", "bestNormalize")

  #install all packages that are not already installed
  install.packages(setdiff(packages, rownames(installed.packages())))


  require(bestNormalize)
  require(nortest)


  BN = bestNormalize(variavel, allow_orderNorm = usar_orderNorm,
                     allow_exp = usar_exponencial, out_of_sample = out_sample,)
  dataframe_variavel_new = BN$x.t
  cat("\n")
  cat("Variável", toupper(var_name), "\n")
  cat("Transformação aplicada", "\n")
  print(BN$chosen_transform)
  cat("\n")
  cat("Testar normalidade", "\n")
  print(ad.test(dataframe_variavel_new))
  amostra = sample(dataframe_variavel_new, 5000, replace = FALSE)
  normalidade = shapiro.test(amostra)
  print(normalidade)
  cat("\n")
  cat("----------------------------------------------------------------------", "\n")
  hist(dataframe_variavel_new, main = var_name)

  return(dataframe_variavel_new)
}



