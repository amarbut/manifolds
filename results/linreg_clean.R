library(tidyverse)
library(corrr)
library(ggplot2)
library(Metrics)
library(stringr)

# metrics <- read_csv("VM_combined_metrics_2.csv")
# glue <- read_csv("VM_combined_glue_2.csv")
metrics <- read_csv("bert_small_metrics.csv")
alt <- read_csv("alt_model_metrics.csv")
roberta <- read_csv("roberta_model_metrics.csv")
base <- read_csv("bert_model_metrics.csv")

colnames(metrics)[1] <- "model"
colnames(metrics)[26] <- "glue_average"
colnames(alt)[1] <- "model"
alt <- alt[,-26]
colnames(roberta)[25] <- "glue_average"
colnames(base)[25] <- "glue_average"

metrics$source = "bert small models"
alt$source = c("bert base models","bert base models","bert base models","bert base models","roberta models","roberta models", "roberta models", "roberta models", "roberta models", "roberta models", "roberta models","bert small models", "bert base models")
roberta$source = "roberta models"
base$source = "bert base models"

combined_models <- metrics%>%
  rbind(roberta)%>%
  rbind(base)%>%
  mutate(source = as.factor(source))

select_metrics <-  c("`p-error`", "`p-iqr`", "`p-skew`", "`p-centroids`", "`p-point_dist`", "`p-patchiness`",
                   "`a-error`", "`a-iqr`", "`a-skew`", "`a-centroids`", "`a-point_dist`", "`a-patchiness`",
                   "`EEE`", "`IsoScore`",
                   "`p-count_var`", "`p-count_kl`", "`a-count_var`", "`a-count_kl`", "`VRM`")
outlier_models <- c("untrained", "rand", "untrained_w_emb", "mse", "mse_filter", "sse", "sse_filter")

alt_lm_df <-  data.frame("model" = alt$model, "glue_average" = alt$glue_average)
lm_list <-  list()

#run alt models through all metrics w/ and w/o model-type info
for (m in select_metrics){
  mod <- lm(paste("glue_average~",m), data = combined_models)
  lm_list[[gsub("`", "", m)]] <- mod
  
  mod_m <- lm(paste("glue_average~",m,"+source"), data = combined_models)
  lm_list[[paste(gsub("`","",m),"_m", sep = "")]] = mod_m
  
  alt_lm_df[gsub("`","",m)] <- predict(mod,alt, se.fit =T)$fit
  alt_lm_df[paste(gsub("`","",m),"_m", sep ="")] <- predict(mod_m,alt,se.fit=T)$fit
  
}

#calculate mse for metrics w/ and w/o model-type and outliers
alt_error_df <- alt_lm_df%>%
  summarise(#across(c(2:30),~sse(glue_average,.), .names = "{col}_sse"),
            across(c(2:40),~mse(glue_average,.), .names = "{col}_mse"),
            #across(c(2:30),~sse(glue_average[!(model %in% outlier_models)],.[!(model %in% outlier_models)]), .names = "{col}_sse_filter"),
            across(c(2:40),~mse(glue_average[!(model %in% outlier_models)],.[!(model %in% outlier_models)]), .names = "{col}_mse_filter"))%>%
  select( -glue_average_mse, -glue_average_mse_filter)%>%
  gather(key = "measure", value = "error")%>%
  mutate(error_type = sub("[^m]+_", "",measure),
         clean_measure = sub("_m.*", "", measure))%>%
  select(-measure)%>%
  spread(key = error_type, value = error)

#combine data for aq_patchiness_(m), aq_point_dist, pq_point_dist for export and viz in python

#TODO: build out df w/ results for combined models for select metrics; concat w/ alt model results; export to csv

combined_results <- combined_models%>%
  select(model, glue_average, type = source, alpha)%>%
  mutate(source = "perturbed_weights", 
         type = ifelse(type == "bert base models", "bert-base",
                       ifelse(type == "bert small models", "bert-small", "roberta")))

combined_results$`a-patchiness` <- predict(lm_list$`a-patchiness`, combined_models, se.fit = T)$fit
combined_results$`a-patchiness_m` <- predict(lm_list$`a-patchiness_m`, combined_models, se.fit = T)$fit
combined_results$`p-point_dist` <- predict(lm_list$`p-point_dist`, combined_models, se.fit = T)$fit
combined_results$`a-point_dist` <- predict(lm_list$`a-point_dist`, combined_models, se.fit = T)$fit
combined_results$`a-point_dist_m` <- predict(lm_list$`a-point_dist_m`, combined_models, se.fit = T)$fit

linreg_data_export <- alt_lm_df%>%
  select(model, glue_average, `a-patchiness`, `a-patchiness_m`, `p-point_dist`, `a-point_dist`, `a-point_dist_m`)%>%
  mutate(type = c("bert-base", "bert-base","bert-base","bert-base","roberta","roberta","roberta","roberta","roberta","roberta","roberta","bert-small","bert-base"),
         source = "alt_models",
         alpha = NA)%>%
  rbind(combined_results)
  
write_csv(linreg_data_export, "linreg_data_export.csv")
  

select_alt <- alt_lm_df%>%
  select(model, glue_average, `p-skew`, `p-patchiness`, `p-centroids`, `p-point_dist`, `a-skew`, `a-patchiness`, `a-centroids`, `a-point_dist`, EEE)
select_results <- alt_error_df%>%
  filter(clean_measure %in% c("p-skew", "p-patchiness", "p-centroids", "p-point_dist", "a-skew", "a-patchiness", "a-centroids", "a-point_dist", "EEE"))
