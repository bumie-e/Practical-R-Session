# Load required libraries
install.packages("tidyverse")
install.packages("janitor")
install.packages("GGally")
install.packages("corrplot")
install.packages("caret")
install.packages(c("pROC","PRROC","ggplot2","ggpubr"))

suppressPackageStartupMessages({
  library(tidyverse)
  library(janitor)
  library(GGally)
  library(corrplot)
  library(caret)
  library(lubridate)
  library(ggplot2)
  library(dplyr)
  library(pROC)
  library(PRROC)
  library(ggpubr)
})



df <- read_csv("Iris.csv")

df <- Iris

# -- basic summaries ----
message("\n== Summary stats ==")
print(df |> summary())

# ---- quick structure & missingness ----
message("Rows: ", nrow(df), " | Cols: ", ncol(df))
print(glimpse(df))

missing_summary <- df |> summarize(across(everything(), ~sum(is.na(.x)))) |> pivot_longer(everything(), names_to="column", values_to="na_count")
print(missing_summary)

# ---- Define target column ----
target_col <- "Species"
df[[target_col]] <- as.factor(df[[target_col]])

print(glimpse(df))

# ---- basic summaries ----
message("\n== Summary stats ==")
print(df |> summary())


# ---- numeric-only correlation (if applicable) ----
num_df <- dplyr::select(df, where(is.numeric))
if (ncol(num_df) >= 2) {
  corr_mat <- cor(num_df, use = "pairwise.complete.obs")
  print(round(corr_mat, 2))
  # visualize correlation
  corrplot::corrplot(corr_mat, method = "circle", type = "lower", tl.cex = 0.8, number.cex = 0.7, addCoef.col = NULL)
}

# ---- pair plot (GGally) ----
if (ncol(num_df) >= 2) {
  ggpairs(bind_cols(num_df, df[target_col]), aes(colour = .data[[target_col]], alpha = 0.7)) +
      theme_minimal() + theme(legend.position = "bottom")
}

# ---- distribution plots for top numeric columns ----
top_num <- names(num_df)[1:min(4, ncol(num_df))]
if (length(top_num) > 0) {
  long_num <- df |> select(all_of(top_num)) |> pivot_longer(everything(), names_to = "feature", values_to = "value")
  p1 <- ggplot(long_num, aes(value)) +
    geom_histogram(bins = 30) +
    facet_wrap(~feature, scales = "free") +
    labs(title = "Distributions of numeric features") +
    theme_minimal()
  print(p1)
}

# ---- boxplots by target (if categorical target exists) ----
if (length(target_col) == 1 && length(top_num) > 0) {
  p2 <- df |>
    pivot_longer(all_of(top_num), names_to="feature", values_to="value") |>
    ggplot(aes(x = .data[[target_col]], y = value)) +
    geom_boxplot(outlier.alpha = 0.4) +
    facet_wrap(~feature, scales = "free_y") +
    labs(title = paste0("Boxplots by ", target_col)) +
    theme_minimal()
  print(p2)
}

# ---- quick model demo (optional): train/test classification if target detected & numeric predictors exist ----
if (length(target_col) == 1 && nlevels(df[[target_col]]) >= 2 && ncol(num_df) >= 2) {
  set.seed(42)
  keep_cols <- c(names(num_df), target_col)
  mdf <- df |> select(all_of(keep_cols)) |> drop_na()
  if (nrow(mdf) >= 50) {
    idx <- createDataPartition(mdf[[target_col]], p = 0.8, list = FALSE)
    train <- mdf[idx, ]; test <- mdf[-idx, ]
    ctrl <- trainControl(method = "cv", number = 5)
    # simple model: kNN (works well for numeric features)
    model <- caret::train(as.formula(paste(target_col, "~ .")),
                          data = train,
                          method = "knn",
                          trControl = ctrl,
                          tuneLength = 5)
    message("\n== kNN model performance (CV) ==")
    print(model)
    preds <- predict(model, newdata = test)
    message("\n== Test confusion matrix ==")
    print(confusionMatrix(preds, test[[target_col]]))
  }
}

message("\nDone. Change `dataset_slug` to point at any Kaggle dataset and re-run.")

# ---- visualising model's results ---

# Build confusion matrix frame
cm <- table(Truth = test[[target_col]], Pred = preds) |> as.data.frame()
colnames(cm) <- c("Truth","Pred","Freq")

ggplot(cm, aes(Pred, Truth, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), fontface = "bold") +
  scale_fill_gradient(low = "#f0f0f0", high = "#2c7fb8") +
  labs(title = "Confusion Matrix (Test Set)", x = "Predicted", y = "Actual") +
  theme_minimal() +
  theme(legend.position = "right")

# --- confidence piority levels ---
# Get per-class probabilities if available
has_probs <- "prob" %in% model$modelType || any(grepl("prob", names(getS3method("predict.train","train"))))
probs <- tryCatch(predict(model, newdata = test, type = "prob"), error = function(e) NULL)

if (!is.null(probs)) {
  # Confidence = max predicted probability
  vis_df <- probs |>
    mutate(.pred_class = colnames(probs)[max.col(probs, ties.method = "first")],
           .max_prob   = apply(probs, 1, max),
           Truth       = test[[target_col]]) |>
    mutate(Correct    = factor(.pred_class == Truth, c(FALSE, TRUE), c("Wrong","Right")))
  
  ggplot(vis_df, aes(.max_prob, fill = Correct)) +
    geom_histogram(bins = 25, position = "identity", alpha = 0.6) +
    facet_wrap(~ Truth, nrow = 1, scales = "free_y") +
    labs(title = "Prediction Confidence by True Class",
         x = "Max predicted probability", y = "Count") +
    theme_minimal()
}

