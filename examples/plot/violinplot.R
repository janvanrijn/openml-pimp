# important for the font: https://github.com/wch/fontcm
# requires once: font_install('fontcm')

library("reshape2")
library("ggplot2")
library("dplyr")
library("extrafont")
library("scales")
loadfonts()

#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

# capitalizes words
simpleCap <- function(x) {
  s <- strsplit(x, " ")[[1]]
  paste(toupper(substring(s, 1,1)), substring(s, 2),
      sep="", collapse=" ")
}

#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

getViolinPlot = function(data, measure = "predictive_accuracy", group = FALSE, 
                         landscape = FALSE, scale01 = FALSE, logscale = FALSE, 
                         sort = 'median', sort_asc = TRUE, font_size = FALSE) {
  
  message(paste("Processing file: ", data, "; measure: ", measure));
  # Reading data
  # Each row in the data frame is a job from study 14 (you can see the data in the csv file)
  data = read.csv(data)
  dodge <- position_dodge(width = 0.8)
  
  # Then I specifiy which measure (column) I want to generate the plot (you can choose another one)
  measure = measure

  # Here I subsample the dataset, selecting just two columns
  #  - flow.name: has the learners names
  #  - measure: the measure I want to analyse
  temp = data[, c("param_name", measure)]
  if (group) {
    temp = data[, c("group", "param_name", measure)]
  }

  # So, now I have a data.frame (matrix) with 2 columns. I will rename them, and will use these names
  #  in the plot call
  if (group) {
    colnames(temp) = c("group", "algo", "meas")
  } else {
    colnames(temp) = c("algo", "meas")
  }
  
  # NEW: ordering algos according to the mean value of the meas.
  factor = 1;
  if (sort_asc == FALSE) {
    factor = -1;
  }
  if (sort == 'mean') {
    aux = lapply(unique(temp$algo), function(lrn){
      obj = na.omit(dplyr::filter(.data = temp, algo == lrn))
      return(factor * mean(obj$meas))
    })
  } else if(sort == 'median') {
    aux = lapply(unique(temp$algo), function(lrn){
      obj = na.omit(dplyr::filter(.data = temp, algo == lrn))
      return(factor * median(obj$meas))
    })
  } else if(sort == 'x_label') {
    aux = lapply(unique(temp$algo), function(lrn){
      return(gsub('.*-([0-9]+).*','\\1',lrn))
    })
  } else {
    stop("Illegal sorting criterium, should be in ('mean','median','x_label')");
  }
  aux = cbind(unique(temp$algo), data.frame(unlist(aux)))
  temp$algo = factor(temp$algo, levels = aux[order(aux$unlist.aux.),1])

  #Create the chart, and map the variable to the axis: algo -> x, meas -> y, fill (color) -> algo
  if (group) {
    # TODO: remove if you don't want this. (jvr)
    levels(temp$algo) <- gsub("; ", "\n", levels(temp$algo))
    g = ggplot(data = temp, mapping = aes(x = as.factor(algo), y = meas, fill = group))
  } else {
    g = ggplot(data = temp, mapping = aes(x = as.factor(algo), y = meas, fill = algo))
  }
  
  
  # Scale y into [0, 1] interval
  if (scale01) {
    g = g + scale_y_continuous(limits = c(0, 1))
  } else if (logscale) {
    g = g + scale_y_log10(breaks=c(0.00390625, 0.015625, 0.0625, 0.25, 1), labels = trans_format("log2", math_format(2^.x)))
  }
  
  # adding the violin
  g = g + geom_violin(trim = TRUE, scale = "width", position = dodge)

  # adding the inner boxplot
  if (group == FALSE) {
    g = g + geom_boxplot(outlier.colour = "black", outlier.size = 0.5, width = 0.2, fill = "white", position = dodge)
  } else { 
    g = g + geom_boxplot(outlier.colour = "black", outlier.size = 0.5, width = 0.2, position = dodge)
  }
  
  # removing the legend, since each learner has a color and a name already identifying them
  if (group == FALSE) {
    g = g + theme(legend.position="none")
  } else {
    g = g + theme(legend.position="bottom",legend.title=element_blank())
  }
  
  g = g + theme(text = element_text(size=20))
  
  # Axis's labels
  g = g + ylab(simpleCap(gsub("_", " ", measure)))

  g = g + theme(axis.text.x = element_text(angle = 45, hjust = 1))

  g = g + theme(text=element_text(family="CM Roman"))
  if (font_size != FALSE) {
    g = g + theme(text=element_text(size=font_size))
  }
  g = g + theme(axis.title.x=element_blank())
  
  
  if (landscape) {
    g = g + coord_flip()
  }
  
  return (g)
}
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

pdf_output = "violin_rf.pdf"
baseclassifiers = getViolinPlot("../../KDD2018/data/fanova/6969/vanilla/ranks_plain_all.csv", "variance_contribution", logscale=FALSE)
ggsave(plot = baseclassifiers, file = pdf_output, width = 8, height = 6, dpi = 600)
embed_fonts(pdf_output, outfile=pdf_output)


pdf_output = "violin_adaboost.pdf"
baseclassifiers = getViolinPlot("../../KDD2018/data/fanova/6970/vanilla/ranks_plain_all.csv", "variance_contribution", logscale=FALSE)
ggsave(plot = baseclassifiers, file = pdf_output, width = 8, height = 6, dpi = 600)
embed_fonts(pdf_output, outfile=pdf_output)


pdf_output = "violin_rbf.pdf"
baseclassifiers = getViolinPlot("../../KDD2018/data/fanova/7707/kernel_rbf/ranks_plain_all.csv", "variance_contribution", logscale=FALSE)
ggsave(plot = baseclassifiers, file = pdf_output, width = 8, height = 6, dpi = 600)
embed_fonts(pdf_output, outfile=pdf_output)


pdf_output = "violin_sigmoid.pdf"
baseclassifiers = getViolinPlot("../../KDD2018/data/fanova/7707/kernel_sigmoid/ranks_plain_all.csv", "variance_contribution", logscale=FALSE)
ggsave(plot = baseclassifiers, file = pdf_output, width = 8, height = 6, dpi = 600)
embed_fonts(pdf_output, outfile=pdf_output)

