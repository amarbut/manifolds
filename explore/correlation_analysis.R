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


#all metrics correlations with glue average
glue_cor <- metrics%>%
  select(-model, -alpha)%>%
  correlate()%>%
  focus(new_average)

#remove multicollinearity
df_nocolin <- metrics%>%
  select(`EEE`, `a-count_var`, `p-count_var`, `p-skew`, `a-skew`, `a-error`, `p-error`, `a-centroids`, `p-centroids`, new_average)

#lm w/ all data points, first order only
m <- lm(new_average~., data = df_nocolin)
summary(m)

#try stepwise model to select features based on AIC
m_select <- step(m,direction = "both",)
summary(m_select)
df_nocolin$pred <- predict(m_select)

#test on alternative model data
alt_select <- alt%>%
  select(`EEE`, `a-count_var`, `p-count_var`, `p-skew`, `a-skew`, `a-error`, `p-error`, `a-centroids`, `p-centroids`, glue_average)

alt_pred <- predict(m_select, alt_select, se.fit = T)
alt <- alt%>%
  mutate(pred_fit = alt_pred$fit)

#test on roberta data
roberta_select <- roberta%>%
  select(`EEE`, `a-count_var`, `p-count_var`, `p-skew`, `a-skew`, `a-error`, `p-error`, `a-centroids`, `p-centroids`, average)

roberta_pred <- predict(m_select, roberta_select, se.fit = T)
roberta <- roberta%>%
  mutate(pred_fit = roberta_pred$fit)

#test on bert data
base_select <- base%>%
  select(`EEE`, `a-count_var`, `p-count_var`, `p-skew`, `a-skew`, `a-error`, `p-error`, `a-centroids`, `p-centroids`, average)

base_pred <- predict(m_select, base_select, se.fit = T)
base <- base%>%
  mutate(pred_fit = base_pred$fit)

#visualize model predictions vs. actual (including alt models)
alt_concat <- alt%>%
  select(glue_average, pred_fit, model)%>%
  mutate(source = "alternative models")

roberta_concat <- roberta%>%
  select(glue_average = average, pred_fit, model)%>%
  mutate(source = "roberta models")

base_concat <- base%>%
  select(glue_average = average, pred_fit, model)%>%
  mutate(source = "bert models")

pred_df <- df_nocolin%>%
  select(glue_average = new_average, pred_fit = pred)%>%
  mutate(source = "perturbed weights", model = metrics$model)%>%
  rbind(alt_concat, roberta_concat, base_concat)

pred_df%>%
  ggplot(aes(x = glue_average, y = pred_fit, color = source))+
  geom_point()+
  geom_abline()+
  geom_text(aes(label=ifelse(source == "alternative models", as.character(model),'')))+
  labs(title = "LinReg Model Fit",
       subtitle = "GLUE~P-Point_dist+P-Count_Var+A-Error+A-Point_Dist+A-Count_Var+EEE+IsoScore",
       x = "Observed GLUE Score",
       y = "Predicted GLUE Score")

#exp with other combinations of variables
#a-count_var alone

acr <- lm(glue_average~`a-count_var`, data = metrics)
metrics$acr_pred = predict(acr)
roberta$acr_pred = predict(acr, roberta, se.fit = T)$fit
base$acr_pred = predict(acr, base, se.fit = T)$fit
alt$acr_pred = predict(acr, alt, se.fit = T)$fit
acr_pred <- metrics%>%
  select(glue_average, acr_pred, model, source)%>% 
  rbind(roberta[c("glue_average", "acr_pred", "model", "source")])%>%
  rbind(base[c("glue_average", "acr_pred", "model", "source")])%>%
  rbind(alt[c("glue_average", "acr_pred", "model", "source")])
  
acr_pred%>%
  ggplot(aes(x = glue_average, y = acr_pred, color = source))+
  geom_point()+
  geom_abline()+
  geom_text(aes(label=ifelse(source == "alternative models", as.character(model),'')))+
  labs(title = "LinReg Model Fit",
       subtitle = "GLUE~A-Count_var",
       x = "Observed GLUE Score",
       y = "Predicted GLUE Score")

#a-count_var with model type
combined_models <- metrics%>%
  rbind(roberta)%>%
  rbind(base)%>%
  mutate(source = as.factor(source))

acr_m <- lm(glue_average~`a-patchiness`+source, data = combined_models)
combined_models$acr_m_pred = predict(acr_m)

alt$acr_m_pred = predict(acr_m, alt, se.fit = T)$fit
acr_m_pred <- combined_models%>%
  select(glue_average, acr_m_pred, model, source)%>% 
  rbind(alt[c("glue_average", "acr_m_pred", "model", "source")])

acr_m_pred%>%
  ggplot(aes(x = glue_average, y = acr_m_pred, color = source))+
  geom_point()+
  geom_abline()+
  # xlim(0,1)+
  # ylim(0,1)+
  geom_text(aes(label=ifelse(is.na(as.numeric(str_sub(model,-1,-1))), as.character(model),'')))+
  labs(title = "LinReg Model Fit",
       subtitle = "GLUE~aq-patchiness+model_type",
       x = "Observed GLUE Score",
       y = "Predicted GLUE Score")

#acr+EEE w/ model
acr_m_EEE <- lm(glue_average~`a-count_var`+source+EEE, data = combined_models)
combined_models$acr_m_EEE_pred = predict(acr_m_EEE)

alt$acr_m_EEE_pred = predict(acr_m_EEE, alt, se.fit = T)$fit
acr_m_EEE_pred <- combined_models%>%
  select(glue_average, acr_m_EEE_pred, model, source)%>% 
  rbind(alt[c("glue_average", "acr_m_EEE_pred", "model", "source")])

acr_m_EEE_pred%>%
  ggplot(aes(x = glue_average, y = acr_m_EEE_pred, color = source))+
  geom_point()+
  geom_abline()+
  geom_text(aes(label=ifelse(is.na(as.numeric(str_sub(model,-1,-1))), as.character(model),'')))+
  labs(title = "LinReg Model Fit",
       subtitle = "GLUE~A-Count_var+EEE+model_type",
       x = "Observed GLUE Score",
       y = "Predicted GLUE Score")

#acr+EEE w/ model & interaction
acr_m_EEE2 <- lm(glue_average~`a-count_var`+source+EEE+I(`a-count_var`*`EEE`), data = combined_models)
combined_models$acr_m_EEE2_pred = predict(acr_m_EEE2)

alt$acr_m_EEE2_pred = predict(acr_m_EEE2, alt, se.fit = T)$fit
acr_m_EEE2_pred <- combined_models%>%
  select(glue_average, acr_m_EEE2_pred, model, source)%>% 
  rbind(alt[c("glue_average", "acr_m_EEE2_pred", "model", "source")])

acr_m_EEE2_pred%>%
  ggplot(aes(x = glue_average, y = acr_m_EEE2_pred, color = source))+
  geom_point()+
  geom_abline()+
  geom_text(aes(label=ifelse(is.na(as.numeric(str_sub(model,-1,-1))), as.character(model),'')))+
  labs(title = "LinReg Model Fit",
       subtitle = "GLUE~A-Count_var+EEE+model_type+(acr*EEE)",
       x = "Observed GLUE Score",
       y = "Predicted GLUE Score")

#acr+a-skew w/ model
acr_m_ask <- lm(glue_average~`a-count_var`+source+`a-skew`, data = combined_models)
combined_models$acr_m_ask_pred = predict(acr_m_ask)

alt$acr_m_ask_pred = predict(acr_m_ask, alt, se.fit = T)$fit
acr_m_ask_pred <- combined_models%>%
  select(glue_average, acr_m_ask_pred, model, source)%>% 
  rbind(alt[c("glue_average", "acr_m_ask_pred", "model", "source")])

acr_m_ask_pred%>%
  ggplot(aes(x = glue_average, y = acr_m_ask_pred, color = source))+
  geom_point()+
  geom_abline()+
  # xlim(0,1)+
  # ylim(0,1)+
  geom_text(aes(label=ifelse(is.na(as.numeric(str_sub(model,-1,-1))), as.character(model),'')))+
  labs(title = "LinReg Model Fit",
       subtitle = "GLUE~A-Count_var+a-skew+model_type",
       x = "Observed GLUE Score",
       y = "Predicted GLUE Score")

#acr+pcr w/ model
acr_m_pcr <- lm(glue_average~`a-count_var`+source+`p-count_var`, data = combined_models)
combined_models$acr_m_pcr_pred = predict(acr_m_pcr)

alt$acr_m_pcr_pred = predict(acr_m_pcr, alt, se.fit = T)$fit
acr_m_pcr_pred <- combined_models%>%
  select(glue_average, acr_m_pcr_pred, model, source)%>% 
  rbind(alt[c("glue_average", "acr_m_pcr_pred", "model", "source")])

acr_m_pcr_pred%>%
  ggplot(aes(x = glue_average, y = acr_m_pcr_pred, color = source))+
  geom_point()+
  geom_abline()+
  # xlim(0,1)+
  # ylim(0,1)+
  geom_text(aes(label=ifelse(is.na(as.numeric(str_sub(model,-1,-1))), as.character(model),'')))+
  labs(title = "LinReg Model Fit",
       subtitle = "GLUE~A-Count_var+p-count_var+model_type",
       x = "Observed GLUE Score",
       y = "Predicted GLUE Score")


#acr+EEE
acr_EEE <- lm(glue_average~`a-count_var`+`EEE`, data = metrics)
metrics$acr_EEE_pred = predict(acr_EEE)
roberta$acr_EEE_pred = predict(acr_EEE, roberta, se.fit = T)$fit
base$acr_EEE_pred = predict(acr_EEE, base, se.fit = T)$fit
alt$acr_EEE_pred = predict(acr_EEE, alt, se.fit = T)$fit
acr_EEE_pred <- metrics%>%
  select(glue_average, acr_EEE_pred, model, source)%>% 
  rbind(roberta[c("glue_average", "acr_EEE_pred", "model", "source")])%>%
  rbind(base[c("glue_average", "acr_EEE_pred", "model", "source")])%>%
  rbind(alt[c("glue_average", "acr_EEE_pred", "model", "source")])

acr_EEE_pred%>%
  ggplot(aes(x = glue_average, y = acr_EEE_pred, color = source))+
  geom_point()+
  geom_abline()+
  geom_text(aes(label=ifelse(source == "alternative models", as.character(model),'')))+
  labs(title = "LinReg Model Fit",
       subtitle = "GLUE~A-Count_var+EEE",
       x = "Observed GLUE Score",
       y = "Predicted GLUE Score")

#acr+ask
acr_ask <- lm(glue_average~`a-count_var`+`a-skew`, data = metrics)
metrics$acr_ask_pred = predict(acr_ask)
roberta$acr_ask_pred = predict(acr_ask, roberta, se.fit = T)$fit
base$acr_ask_pred = predict(acr_ask, base, se.fit = T)$fit
alt$acr_ask_pred = predict(acr_ask, alt, se.fit = T)$fit
acr_ask_pred <- metrics%>%
  select(glue_average, acr_ask_pred, model, source)%>% 
  rbind(roberta[c("glue_average", "acr_ask_pred", "model", "source")])%>%
  rbind(base[c("glue_average", "acr_ask_pred", "model", "source")])%>%
  rbind(alt[c("glue_average", "acr_ask_pred", "model", "source")])

acr_ask_pred%>%
  ggplot(aes(x = glue_average, y = acr_ask_pred, color = source))+
  geom_point()+
  geom_abline()+
  geom_text(aes(label=ifelse(source == "alternative models", as.character(model),'')))+
  labs(title = "LinReg Model Fit",
       subtitle = "GLUE~A-Count_var+a-skew",
       x = "Observed GLUE Score",
       y = "Predicted GLUE Score")

#acr+pcr
acr_pcr <- lm(glue_average~`a-count_var`+`p-count_var`, data = metrics)
metrics$acr_pcr_pred = predict(acr_pcr)
roberta$acr_pcr_pred = predict(acr_pcr, roberta, se.fit = T)$fit
base$acr_pcr_pred = predict(acr_pcr, base, se.fit = T)$fit
alt$acr_pcr_pred = predict(acr_pcr, alt, se.fit = T)$fit
acr_pcr_pred <- metrics%>%
  select(glue_average, acr_pcr_pred, model, source)%>% 
  rbind(roberta[c("glue_average", "acr_pcr_pred", "model", "source")])%>%
  rbind(base[c("glue_average", "acr_pcr_pred", "model", "source")])%>%
  rbind(alt[c("glue_average", "acr_pcr_pred", "model", "source")])

acr_pcr_pred%>%
  ggplot(aes(x = glue_average, y = acr_pcr_pred, color = source))+
  geom_point()+
  geom_abline()+
  geom_text(aes(label=ifelse(source == "alternative models", as.character(model),'')))+
  labs(title = "LinReg Model Fit",
       subtitle = "GLUE~A-Count_var+p-count_var",
       x = "Observed GLUE Score",
       y = "Predicted GLUE Score")

#acr+pcr+ask
acr_ask_pcr <- lm(glue_average~`a-count_var`+`a-skew`+`p-count_var`, data = metrics)
metrics$acr_ask_pcr_pred = predict(acr_ask_pcr)
roberta$acr_ask_pcr_pred = predict(acr_ask_pcr, roberta, se.fit = T)$fit
base$acr_ask_pcr_pred = predict(acr_ask_pcr, base, se.fit = T)$fit
alt$acr_ask_pcr_pred = predict(acr_ask_pcr, alt, se.fit = T)$fit
acr_ask_pcr_pred <- metrics%>%
  select(glue_average, acr_ask_pcr_pred, model, source)%>% 
  rbind(roberta[c("glue_average", "acr_ask_pcr_pred", "model", "source")])%>%
  rbind(base[c("glue_average", "acr_ask_pcr_pred", "model", "source")])%>%
  rbind(alt[c("glue_average", "acr_ask_pcr_pred", "model", "source")])

acr_ask_pcr_pred%>%
  ggplot(aes(x = glue_average, y = acr_ask_pcr_pred, color = source))+
  geom_point()+
  geom_abline()+
  xlim(0.25,1)+
  ylim(0.25,1)+
  geom_text(aes(label=ifelse(source == "alternative models", as.character(model),'')))+
  labs(title = "LinReg Model Fit",
       subtitle = "GLUE~A-Count_var+a-skew+p-count_var",
       x = "Observed GLUE Score",
       y = "Predicted GLUE Score")

#acr+pcr+ask+EEE
acr_ask_pcr_EEE_pcr <- lm(glue_average~`a-count_var`+`a-skew`+`p-count_var`+`EEE`, data = metrics)
metrics$acr_ask_pcr_EEE_pcr_pred = predict(acr_ask_pcr_EEE_pcr)
roberta$acr_ask_pcr_EEE_pcr_pred = predict(acr_ask_pcr_EEE_pcr, roberta, se.fit = T)$fit
base$acr_ask_pcr_EEE_pcr_pred = predict(acr_ask_pcr_EEE_pcr, base, se.fit = T)$fit
alt$acr_ask_pcr_EEE_pcr_pred = predict(acr_ask_pcr_EEE_pcr, alt, se.fit = T)$fit
acr_ask_pcr_EEE_pcr_pred <- metrics%>%
  select(glue_average, acr_ask_pcr_EEE_pcr_pred, model, source)%>% 
  rbind(roberta[c("glue_average", "acr_ask_pcr_EEE_pcr_pred", "model", "source")])%>%
  rbind(base[c("glue_average", "acr_ask_pcr_EEE_pcr_pred", "model", "source")])%>%
  rbind(alt[c("glue_average", "acr_ask_pcr_EEE_pcr_pred", "model", "source")])

acr_ask_pcr_EEE_pcr_pred%>%
  ggplot(aes(x = glue_average, y = acr_ask_pcr_EEE_pcr_pred, color = source))+
  geom_point()+
  geom_abline()+
  xlim(0.25,1)+
  ylim(0.25,1)+
  geom_text(aes(label=ifelse(source == "alternative models", as.character(model),'')))+
  labs(title = "LinReg Model Fit",
       subtitle = "GLUE~A-Count_var+a-skew+p-count_var+EEE",
       x = "Observed GLUE Score",
       y = "Predicted GLUE Score")

#pcr
pcr <- lm(glue_average~`p-count_var`, data = metrics)
metrics$pcr_pred = predict(pcr)
roberta$pcr_pred = predict(pcr, roberta, se.fit = T)$fit
base$pcr_pred = predict(pcr, base, se.fit = T)$fit
alt$pcr_pred = predict(pcr, alt, se.fit = T)$fit
pcr_pred <- metrics%>%
  select(glue_average, pcr_pred, model, source)%>% 
  rbind(roberta[c("glue_average", "pcr_pred", "model", "source")])%>%
  rbind(base[c("glue_average", "pcr_pred", "model", "source")])%>%
  rbind(alt[c("glue_average", "pcr_pred", "model", "source")])

pcr_pred%>%
  ggplot(aes(x = glue_average, y = pcr_pred, color = source))+
  geom_point()+
  geom_abline()+
  xlim(0.25,1)+
  ylim(0.25,1)+
  geom_text(aes(label=ifelse(source == "alternative models", as.character(model),'')))+
  labs(title = "LinReg Model Fit",
       subtitle = "GLUE~p-count_var",
       x = "Observed GLUE Score",
       y = "Predicted GLUE Score")

#pcr+EEE
pcr_EEE <- lm(glue_average~`p-count_var`+`EEE`, data = metrics)
metrics$pcr_EEE_pred = predict(pcr_EEE)
roberta$pcr_EEE_pred = predict(pcr_EEE, roberta, se.fit = T)$fit
base$pcr_EEE_pred = predict(pcr_EEE, base, se.fit = T)$fit
alt$pcr_EEE_pred = predict(pcr_EEE, alt, se.fit = T)$fit
pcr_EEE_pred <- metrics%>%
  select(glue_average, pcr_EEE_pred, model, source)%>% 
  rbind(roberta[c("glue_average", "pcr_EEE_pred", "model", "source")])%>%
  rbind(base[c("glue_average", "pcr_EEE_pred", "model", "source")])%>%
  rbind(alt[c("glue_average", "pcr_EEE_pred", "model", "source")])

pcr_EEE_pred%>%
  ggplot(aes(x = glue_average, y = pcr_EEE_pred, color = source))+
  geom_point()+
  geom_abline()+
  xlim(0.25,1)+
  ylim(0.25,1)+
  geom_text(aes(label=ifelse(source == "alternative models", as.character(model),'')))+
  labs(title = "LinReg Model Fit",
       subtitle = "GLUE~p-count_var+EEE",
       x = "Observed GLUE Score",
       y = "Predicted GLUE Score")

#pcr+ask
pcr_ask <- lm(glue_average~`p-count_var`+`a-skew`, data = metrics)
metrics$pcr_ask_pred = predict(pcr_ask)
roberta$pcr_ask_pred = predict(pcr_ask, roberta, se.fit = T)$fit
base$pcr_ask_pred = predict(pcr_ask, base, se.fit = T)$fit
alt$pcr_ask_pred = predict(pcr_ask, alt, se.fit = T)$fit
pcr_ask_pred <- metrics%>%
  select(glue_average, pcr_ask_pred, model, source)%>% 
  rbind(roberta[c("glue_average", "pcr_ask_pred", "model", "source")])%>%
  rbind(base[c("glue_average", "pcr_ask_pred", "model", "source")])%>%
  rbind(alt[c("glue_average", "pcr_ask_pred", "model", "source")])

pcr_ask_pred%>%
  ggplot(aes(x = glue_average, y = pcr_ask_pred, color = source))+
  geom_point()+
  geom_abline()+
  xlim(0.25,1)+
  ylim(0.25,1)+
  geom_text(aes(label=ifelse(source == "alternative models", as.character(model),'')))+
  labs(title = "LinReg Model Fit",
       subtitle = "GLUE~p-count_var+a-skew",
       x = "Observed GLUE Score",
       y = "Predicted GLUE Score")

#pcr+EEE+ask
pcr_EEE_ask <- lm(glue_average~`p-count_var`+`EEE`+`a-skew`, data = metrics)
metrics$pcr_EEE_ask_pred = predict(pcr_EEE_ask)
roberta$pcr_EEE_ask_pred = predict(pcr_EEE_ask, roberta, se.fit = T)$fit
base$pcr_EEE_ask_pred = predict(pcr_EEE_ask, base, se.fit = T)$fit
alt$pcr_EEE_ask_pred = predict(pcr_EEE_ask, alt, se.fit = T)$fit
pcr_EEE_ask_pred <- metrics%>%
  select(glue_average, pcr_EEE_ask_pred, model, source)%>% 
  rbind(roberta[c("glue_average", "pcr_EEE_ask_pred", "model", "source")])%>%
  rbind(base[c("glue_average", "pcr_EEE_ask_pred", "model", "source")])%>%
  rbind(alt[c("glue_average", "pcr_EEE_ask_pred", "model", "source")])

pcr_EEE_ask_pred%>%
  ggplot(aes(x = glue_average, y = pcr_EEE_ask_pred, color = source))+
  geom_point()+
  geom_abline()+
  xlim(0.25,1)+
  ylim(0.25,1)+
  geom_text(aes(label=ifelse(source == "alternative models", as.character(model),'')))+
  labs(title = "LinReg Model Fit",
       subtitle = "GLUE~p-count_var+EEE+a-skew",
       x = "Observed GLUE Score",
       y = "Predicted GLUE Score")

#try higher order pcr terms
#pcr+pcr^2
pcr2 <- lm(glue_average~`p-count_var`+I(`p-count_var`**2), data = metrics)
metrics$pcr2_pred = predict(pcr2)
roberta$pcr2_pred = predict(pcr2, roberta, se.fit = T)$fit
base$pcr2_pred = predict(pcr2, base, se.fit = T)$fit
alt$pcr2_pred = predict(pcr2, alt, se.fit = T)$fit
pcr2_pred <- metrics%>%
  select(glue_average, pcr2_pred, model, source)%>% 
  rbind(roberta[c("glue_average", "pcr2_pred", "model", "source")])%>%
  rbind(base[c("glue_average", "pcr2_pred", "model", "source")])%>%
  rbind(alt[c("glue_average", "pcr2_pred", "model", "source")])

pcr2_pred%>%
  ggplot(aes(x = glue_average, y = pcr2_pred, color = source))+
  geom_point()+
  geom_abline()+
  xlim(0.25,1)+
  ylim(0.25,1)+
  geom_text(aes(label=ifelse(source == "alternative models", as.character(model),'')))+
  labs(title = "LinReg Model Fit",
       subtitle = "GLUE~p-count_var+p-count_var^2",
       x = "Observed GLUE Score",
       y = "Predicted GLUE Score")

#pcr+pcr^2+pcr^3
pcr3 <- lm(glue_average~`p-count_var`+I(`p-count_var`**2)+`a-skew`, data = metrics)
metrics$pcr3_pred = predict(pcr3)
roberta$pcr3_pred = predict(pcr3, roberta, se.fit = T)$fit
base$pcr3_pred = predict(pcr3, base, se.fit = T)$fit
alt$pcr3_pred = predict(pcr3, alt, se.fit = T)$fit
pcr3_pred <- metrics%>%
  select(glue_average, pcr3_pred, model, source)%>% 
  rbind(roberta[c("glue_average", "pcr3_pred", "model", "source")])%>%
  rbind(base[c("glue_average", "pcr3_pred", "model", "source")])%>%
  rbind(alt[c("glue_average", "pcr3_pred", "model", "source")])

pcr3_pred%>%
  ggplot(aes(x = glue_average, y = pcr3_pred, color = source))+
  geom_point()+
  geom_abline()+
  xlim(0.25,1)+
  ylim(0.25,1)+
  geom_text(aes(label=ifelse(source == "alternative models", as.character(model),'')))+
  labs(title = "LinReg Model Fit",
       subtitle = "GLUE~p-count_var+p-count_var^2+p-count_var^3",
       x = "Observed GLUE Score",
       y = "Predicted GLUE Score")

#EEE
EEE <- lm(glue_average~`EEE`, data = metrics)
metrics$EEE_pred = predict(EEE)
roberta$EEE_pred = predict(EEE, roberta, se.fit = T)$fit
base$EEE_pred = predict(EEE, base, se.fit = T)$fit
alt$EEE_pred = predict(EEE, alt, se.fit = T)$fit
EEE_pred <- metrics%>%
  select(glue_average, EEE_pred, model, source)%>% 
  rbind(roberta[c("glue_average", "EEE_pred", "model", "source")])%>%
  rbind(base[c("glue_average", "EEE_pred", "model", "source")])%>%
  rbind(alt[c("glue_average", "EEE_pred", "model", "source")])

EEE_pred%>%
  ggplot(aes(x = glue_average, y = EEE_pred, color = source))+
  geom_point()+
  geom_abline()+
  xlim(0.25,1)+
  ylim(0.25,1)+
  geom_text(aes(label=ifelse(is.na(as.numeric(str_sub(model,-1,-1))), as.character(model),'')))+
  labs(title = "LinReg Model Fit",
       subtitle = "GLUE~EEE",
       x = "Observed GLUE Score",
       y = "Predicted GLUE Score")

#EEE+model
EEE_m <- lm(glue_average~`EEE`+source+`a-skew`, data = combined_models)
combined_models$EEE_m_pred = predict(EEE_m)

alt$EEE_m_pred = predict(EEE_m, alt, se.fit = T)$fit
EEE_m_pred <- combined_models%>%
  select(glue_average, EEE_m_pred, model, source)%>% 
  rbind(alt[c("glue_average", "EEE_m_pred", "model", "source")])

EEE_m_pred%>%
  ggplot(aes(x = glue_average, y = EEE_m_pred, color = source))+
  geom_point()+
  geom_abline()+
  # xlim(0,1)+
  # ylim(0,1)+
  geom_text(aes(label=ifelse(is.na(as.numeric(str_sub(model,-1,-1))), as.character(model),'')))+
  labs(title = "LinReg Model Fit",
       subtitle = "GLUE~EEE+model_type",
       x = "Observed GLUE Score",
       y = "Predicted GLUE Score")

#compare alt model pred errors
error <- alt%>%
  select(model,glue_average, c(42:45))%>%
  filter(model != "untrained_w_emb", model != "rand")%>%
  #mutate(across(c(3:13), ~.-glue_average, .names = "{col}_error"))#%>%
  summarise(across(c(2:6),~sse(.,glue_average), .names = "{col}_sse"),
           across(c(2:6),~mse(.,glue_average), .names = "{col}_mse"))
