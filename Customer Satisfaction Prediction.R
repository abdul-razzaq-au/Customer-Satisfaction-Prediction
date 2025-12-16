#Data pre processing

## remove hash before every line below to install packages
#used #to avoid reinstalling everytime
#install.packages("tidyverse")
#install.packages("lubridate")
#install.packages("janitor")
#install.packages("skimr")
#install.packages("arrow")
#install.packages("tidytext")
#install.packages("textdata")
#install.packages("tm")
#install.packages("topicmodels")
#install.packages("caret")
#install.packages("ranger")
#install.packages("ggplot2")
#install.packages("data.table")
#install.packages("plotly")    # optional interactive viz

library(tidyverse)
library(lubridate)
library(janitor)
library(skimr)
library(arrow)
library(dplyr)
library(tidyverse)
library(ggplot2)
library(reshape2)
library(cowplot)
library(ranger)
library(caret)

# Load data
df <- read_csv("C:/Users/abdul/Downloads/customer_support_tickets.csv", show_col_types = FALSE) %>%
  clean_names()

# Basic info

colSums(is.na(df))
print(dim(df))
print(names(df))
skim(df)
head(df)

# Date conversions
df <- df %>%
  mutate(
    date_of_purchase = parse_date_time(date_of_purchase, orders = c("ymd","mdy","dmy")),
    first_response_time = parse_date_time(first_response_time, orders = c("ymd HMS","mdy HMS")),
    time_to_resolution = parse_date_time(time_to_resolution, orders = c("ymd HMS","mdy HMS"))
  )

# Handle missing values
df <- df %>% drop_na(customer_satisfaction_rating)

# Encode categorical values as factors
df <- df %>% mutate_if(is.character, as.factor)

# Save cleaned file
write_parquet(df, "C:/Users/abdul/Downloads/cleaned_data.parquet")
write_csv(df, "C:/Users/abdul/Downloads/cleaned_data.csv")


print("Preprocessing Completed!")

#EDA

df <- read_parquet("C:/Users/abdul/Downloads/cleaned_data.parquet")

# 1. Distribution of satisfaction
ggplot(df, aes(customer_satisfaction_rating)) +
  geom_bar(fill="skyblue") +
  labs(title="Satisfaction Rating Distribution")


# 2. Ticket types
ggplot(df, aes(ticket_type)) +
  geom_bar(fill="orange") +
  coord_flip() +
  labs(title="Ticket Type Count")

#3. Ticket Priority Count
# ================================
ggplot(df, aes(ticket_priority)) +
  geom_bar(fill="#7B92AA") +
  labs(title="Ticket Priority Distribution")

#4. Ticket Channel Count
# ================================
ggplot(df, aes(ticket_channel)) +
  geom_bar(fill="#50E3C2") +
  coord_flip() +
  labs(title="Ticket Channel Distribution")

#5. Customer Age Distribution (Line Chart)
# ================================
age_counts <- df %>%
  group_by(customer_age) %>%
  summarise(count = n()) %>%
  filter(!is.na(customer_age))

ggplot(age_counts, aes(x=customer_age, y=count)) +
  geom_line(color="#BD10E0", size=0.5) +
  geom_point(color="#BD10E0", size=1.5) +
  labs(title="Customer Age Distribution (Line Chart)",
       x="Age", y="Number of Customers")

#6. Boxplot: Satisfaction by Ticket Priority
# ================================
ggplot(df, aes(x=ticket_priority, y=customer_satisfaction_rating)) +
  geom_boxplot(fill="#F8E71C") +
  labs(title="Satisfaction Rating by Ticket Priority",
       x="Ticket Priority", y="Satisfaction Rating")

#Feature Engineering

df <- read_parquet("C:/Users/abdul/Downloads/cleaned_data.parquet")

#engineer the data
df <- df %>%
  mutate(
    first_response_secs = as.numeric(difftime(first_response_time, date_of_purchase, units="secs")),
    resolution_secs = as.numeric(difftime(time_to_resolution, date_of_purchase, units="secs")),
    satisfaction_binary = if_else(customer_satisfaction_rating >= 4, 1, 0)
  )

write_parquet(df, "C:/Users/abdul/Downloads/engineered_data.parquet")     #saves data in parquet format
write_csv(df, "C:/Users/abdul/Downloads/engineered_data.csv")             #saves the data in csv format

#Preview the engineered data
engineered_data <- read.csv("C:/Users/abdul/Downloads/engineered_data.csv")
summary(engineered_data)


# DEFINE X (FEATURE SET) & Y (TARGET)

# y → target variable
y <- df$satisfaction_binary

# X → feature set
X <- df %>%
  select(
    customer_age,
    ticket_type,
    ticket_priority,
    ticket_channel,
    first_response_secs,
    resolution_secs
  )

# Convert categorical features to factors
X <- X %>% mutate_if(is.character, as.factor)

# Combine X and y for splitting
df_model <- cbind(X, satisfaction_binary = y)


# TRAIN-TEST SPLIT
set.seed(123)

trainIndex <- createDataPartition(df_model$satisfaction_binary, p = 0.8, list = FALSE)

train <- df_model[trainIndex, ]
test  <- df_model[-trainIndex, ]

write_parquet(train, "C:/Users/abdul/Downloads/train_data.parquet")
write_parquet(test, "C:/Users/abdul/Downloads/test_data.parquet")

write_csv(train, "C:/Users/abdul/Downloads/train_data.csv")
write_csv(test, "C:/Users/abdul/Downloads/test_data.csv")

#preview train set and test set

train_set <- read.csv("C:/Users/abdul/Downloads/train_data.csv")
test_set <-  read.csv("C:/Users/abdul/Downloads/test_data.csv")

head(train_set)
summary(train_set)

head(test_set)
summary(test_set)

print("Feature engineering and train-test split completed!")

#Model building

# Load TRAIN dataset
train <- read_csv("C:/Users/abdul/Downloads/train_data.csv", show_col_types = FALSE)

# Convert character columns to factors

train <- train %>% mutate_if(is.character, as.factor)

# Convert target to factor with valid names
train$satisfaction_binary <- factor(
  train$satisfaction_binary,
  levels = c(0, 1),
  labels = c("Dissatisfied", "Satisfied")
)

# Define feature columns
feature_cols <- c(
  "customer_age",
  "ticket_type",
  "ticket_priority",
  "ticket_channel",
  "first_response_secs",
  "resolution_secs"
)

numeric_cols <- c("customer_age", "first_response_secs", "resolution_secs")

#Extract X and y
X_train <- train %>% select(all_of(feature_cols))
y_train <- train$satisfaction_binary

#Fit scaler on training NUMERIC columns only

pre_proc <- preProcess(X_train[, numeric_cols], method = c("center", "scale"))

X_train_scaled <- X_train
X_train_scaled[, numeric_cols] <- predict(pre_proc, X_train[, numeric_cols])

# -----------------------------
#Combine scaled X with y
# -----------------------------
train_final <- cbind(X_train_scaled, satisfaction_binary = y_train)

# -----------------------------
#Train Random Forest model
# -----------------------------
set.seed(123)
model <- train(
  satisfaction_binary ~ .,
  data = train_final,
  method = "ranger",
  trControl = trainControl(
    method = "cv",
    number = 5,
    classProbs = TRUE,
    summaryFunction = twoClassSummary
  ),
  metric = "ROC",
  importance = "impurity"
)


#Save model and scaler
dir.create("models", showWarnings = FALSE)
saveRDS(model, "C:/Users/abdul/Downloads/satisfaction_model.rds")
saveRDS(pre_proc, "C:/Users/abdul/Downloads//preproc.rds")

print("Model training completed successfully!")

#_Model Evaluation

# Load model + scaler
model <- readRDS("C:/Users/abdul/Downloads/satisfaction_model.rds")
pre_proc <- readRDS("C:/Users/abdul/Downloads/preproc.rds")

# Load TEST dataset
test <- read_csv("C:/Users/abdul/Downloads/test_data.csv", show_col_types = FALSE)
test <- test %>% mutate_if(is.character, as.factor)


#Convert target to factor
test$satisfaction_binary <- factor(
  test$satisfaction_binary,
  levels = c(0, 1),
  labels = c("Dissatisfied", "Satisfied")
)

#Select features
feature_cols <- c(
  "customer_age",
  "ticket_type",
  "ticket_priority",
  "ticket_channel",
  "first_response_secs",
  "resolution_secs"
)

numeric_cols <- c("customer_age", "first_response_secs","resolution_secs")

# Extract X_test and y_test
X_test <- test %>% select(all_of(feature_cols))
y_test <- test$satisfaction_binary


#Apply SAME SCALING as training
X_test_scaled <- X_test
X_test_scaled[, numeric_cols] <- predict(pre_proc, X_test[, numeric_cols])

#Predict
# -----------------------------
pred_class <- predict(model, X_test_scaled)
pred_prob <- predict(model, X_test_scaled, type = "prob")

#Evaluate
# -----------------------------
conf <- confusionMatrix(pred_class, y_test)
print(conf)

print("Model evaluation completed successfully!")
