# ============================================
# Online Retail (Kaggle) → Association Rules
# ============================================

install.packages("arules")
install.packages("arulesViz")

suppressPackageStartupMessages({
  library(tidyverse)
  library(janitor)
  library(lubridate)
  library(arules)
  library(arulesViz)
})

# ----------------------------
# 0) Config
# ----------------------------
dataset_slug <- "vijayuv/onlineretail"   # Kaggle dataset slug
work_dir     <- file.path(tempdir(), "kaggle_onlineretail")
dir.create(work_dir, recursive = TRUE, showWarnings = FALSE)

# -----------------------
# 2) Load & Clean
# ----------------------------
raw <- read_csv("data.csv")

# Expect typical columns: InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country
# Basic hygiene for transactional mining:
df <- raw |>
  mutate(
    invoice_date = suppressWarnings(lubridate::dmy_hms(invoice_date)) %||% lubridate::parse_date_time(invoice_date, orders = c("mdy HMS", "dmy HMS", "mdy HM", "dmy HM")),
    invoice_no   = as.character(invoice_no),
    description  = str_squish(str_to_title(description)),
    country      = as.factor(country)
  ) |>
  # Remove cancellations/credits (InvoiceNo starting with 'C'), missing descriptions, and non-positive quantities/prices
  filter(!is.na(description),
         !str_detect(invoice_no, "^C"),
         quantity > 0,
         unit_price > 0)

# Optional: focus on a single market (reduces sparsity & runtime)
# Most analyses use the UK subset due to size and consistency.
df <- df |> filter(country == "United Kingdom")

# Optional: limit to a date window (the dataset spans ~Dec 2010–Dec 2011)
# df <- df |> filter(invoice_date >= as_datetime("2011-01-01"), invoice_date < as_datetime("2011-07-01"))

message("Rows after filtering: ", nrow(df))
stopifnot(nrow(df) > 0)

# ----------------------------
# 3) Build transactions (basket = InvoiceNo)
# ----------------------------
# We'll use product Description as the item label (human-readable).
basket_tbl <- df |>
  select(invoice_no, description) |>
  distinct()                       # avoid duplicates of same item in same invoice

# Convert to "transactions" object
# arules accepts a transactions object via split or read.transactions with a format.
tx_list <- split(basket_tbl$description, basket_tbl$invoice_no)    # list: invoice -> items
tx <- as(tx_list, "transactions")

summary(tx)
itemFrequencyPlot(tx, topN = 20, type = "absolute",
                  main = "Top 20 Items (Absolute Frequency)")

# ----------------------------
# 4) Mine frequent itemsets (Apriori)
# ----------------------------
# Choose thresholds (tune for size/speed).
# Start conservative to get meaningful, not too many rules:
min_support   <- 0.003   # ~0.3% of baskets
min_confidence<- 0.25
min_len       <- 2

itemsets <- apriori(tx,
                    parameter = list(target = "frequent",
                                     supp = min_support,
                                     minlen = min_len))
inspect(head(sort(itemsets, by = "support"), 10))

# ----------------------------
# 5) Generate association rules
# ----------------------------
rules <- apriori(tx,
                 parameter = list(supp = min_support,
                                  conf = min_confidence,
                                  minlen = min_len,
                                  target = "rules"))

# Basic diagnostics
summary(rules)
length(rules)

# Sort by lift and view top rules
top_rules <- head(sort(rules, by = "lift", decreasing = TRUE), 20)
inspect(top_rules)

# Remove redundant rules (keeps concise, non-dominated patterns)
nonred_rules <- rules[!is.redundant(rules)]
message("Rules before/after redundancy filter: ", length(rules), " → ", length(nonred_rules))

# Filter for stronger/shorter rules (optional)
strong_rules <- subset(nonred_rules, lift > 3 & confidence > 0.35)
inspect(head(sort(strong_rules, by = "lift"), 15))

# ----------------------------
# 6) Visualize rules (arulesViz)
# ----------------------------
# NOTE: Some plots open in an interactive device; if running non-interactively, use plot(..., engine="htmlwidget")
# Matrix / Heatmap view
plot(nonred_rules, method = "matrix", measure = "lift", control = list(reorder = TRUE))

# Graph network of top rules (limit to avoid clutter)
plot(head(sort(nonred_rules, by = "lift"), 50), method = "graph")

# Two-key plot (support vs confidence, color = lift)
plot(nonred_rules, measure = c("support","confidence"), shading = "lift")

# Parallel coordinate plot (antecedent→consequent flows)
plot(head(sort(nonred_rules, by = "lift"), 30), method = "paracoord")

# Save a static plot to file (example)
png(file.path(work_dir, "two_key_plot.png"), width = 1200, height = 900, res = 150)
plot(nonred_rules, measure = c("support","confidence"), shading = "lift")
dev.off()

# ----------------------------
# 7) Interpreting & exporting
# ----------------------------
# Tidy the rules to a data frame with quality measures
rules_df <- as(nonred_rules, "data.frame") |>
  arrange(desc(lift))

# Peek at the top 10
print(head(rules_df, 10))

# Export CSV
out_csv <- file.path(work_dir, "association_rules.csv")
readr::write_csv(rules_df, out_csv)
message("Exported rules to: ", out_csv)

# ----------------------------
# 8) (Optional) Drill-downs
# ----------------------------
# a) Focus on baskets containing a specific popular item
popular_items <- names(sort(itemFrequency(tx), decreasing = TRUE))[1:10]
focus_item <- popular_items[1]
message("Focusing on item: ", focus_item)

rules_focus <- subset(nonred_rules, lhs %in% focus_item | rhs %in% focus_item)
inspect(head(sort(rules_focus, by = "lift"), 10))
plot(head(sort(rules_focus, by = "lift"), 20), method = "graph")

# b) Mine bigger bundles (triplets+) for cross-sell packs
triplet_rules <- apriori(tx, parameter = list(supp = min_support,
                                              conf = min_confidence,
                                              minlen = 3, maxlen = 4))
triplet_rules <- triplet_rules[!is.redundant(triplet_rules)]
inspect(head(sort(triplet_rules, by = "lift"), 10))

message("\nDone. Adjust support/confidence/minlen for more/less rules.")
