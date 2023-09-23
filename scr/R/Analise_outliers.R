# Script para a análise de outliers
# Autor: André Rizzo
# Data: 20/04/2023
# Verão: 1.0
# Referência: https://statsandr.com/blog/outliers-detection-in-r/#references



# Teste de Rosner
detecta_outlier_teste_rosner = function(dataframe, variavel, dataframe_variavel,
                                        uso_orderNorm, uso_exponencial,
                                        uso_outsample) {

  # Define packages to install
  packages <- c("EnvStats", "bestNormalize")

  # Install all packages that are not already installed
  install.packages(setdiff(packages, rownames(installed.packages())))

  library(EnvStats)
  library(bestNormalize)

  amostra = sample(dataframe_variavel, 5000, replace = FALSE)
  normalidade = shapiro.test(amostra)

  if (normalidade$p.value <= 0.05) {
    # Os dados não seguem uma distribuição Normal
    # Verifica a transformaçao mais adequada
    cat("Os dados não são normalmente distribuídos.", "\n")
    cat("\n")
    var_transformada = transforma_variavel(dataframe,
                                           variavel,
                                           dataframe_variavel,
                                           usar_orderNorm = uso_orderNorm,
                                           usar_exponencial = uso_exponencial,
                                           out_sample = uso_outsample)
    variavel = var_transformada

  } else if (normalidade$p.value > 0.05) {
    status = "Os dados são normalmente distribuídos."
    cat(satus)
    cat("\n")
  }
  resultado_rosner = rosnerTest(variavel, k = 5)
  print(resultado_rosner$all.stats)
  cat("\n")
}


transforma_variavel = function(dataframe, variavel, dataframe_variavel,
                               usar_orderNorm, usar_exponencial, out_sample) {

  # Define packages to install
  packages <- c("nortest", "bestNormalize")

  # Install all packages that are not already installed
  install.packages(setdiff(packages, rownames(installed.packages())))

  require(bestNormalize)
  require(nortest)


  BN = bestNormalize(dataframe_variavel, allow_orderNorm = usar_orderNorm,
                     allow_exp = usar_exponencial, out_of_sample = out_sample,)
  dataframe_variavel_new = BN$x.t
  cat("\n")
  cat("Variável", toupper(variavel), "\n")
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
  hist(dataframe_variavel_new, main = variavel)

  return(dataframe_variavel_new)
}


detecta_outlier_teste_rosner(dataframe = df,
                             variavel = "price",
                             dataframe_variavel = df$price,
                             uso_orderNorm = TRUE,
                             uso_exponencial = FALSE,
                             uso_outsample = FALSE)


