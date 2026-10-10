library(tm)
library(SnowballC)
library(caTools)
library(e1071)
library(caret)
library(glmnet)
library(ggplot2)
data <- read.csv("C:/dataset/spam.csv", stringsAsFactors = FALSE, fileEncoding = "latin1")
data <- data[, c("v1", "v2")]
colnames(data) <- c("label", "text")

str(data)
table(data$label)
summary(data)

sum(is.na(data$v2))
sum(is.na(data$v1))

data$msg_length <- nchar(data$text)

length_histogram <- ggplot(data, aes(x = msg_length, fill = label)) +
  geom_histogram(binwidth = 5, position = "identity", alpha = 0.7) + 
  scale_fill_manual(values = c("ham" = "#4CAF50", "spam" = "#F44336")) +
  labs(
    title = "Message Length Distribution: Spam vs. Ham", 
    x = "Number of Characters", 
    y = "Frequency"
  ) +
  coord_cartesian(xlim = c(0, 250)) + 
  theme_minimal()

print(length_histogram)
data$msg_length <- NULL

data$text <- iconv(data$text, from = "latin1", to = "ASCII", sub = " ")  
data$text <- gsub("[^a-zA-Z ]", " ", data$text)                        
data$text <- gsub("\\s+", " ", data$text)                             

corpus <- VCorpus(VectorSource(data$text))
corpus_preprocess <- tm_map(corpus, content_transformer(tolower))
corpus_preprocess <- tm_map(corpus_preprocess, removeWords, stopwords("en"))
corpus_preprocess <- tm_map(corpus_preprocess, stemDocument)
corpus_preprocess <- tm_map(corpus_preprocess, stripWhitespace)

dtm_bow <- DocumentTermMatrix(corpus_preprocess)
head(Terms(dtm_bow), 20)
inspect(dtm_bow[1:5, 1:10])

set.seed(42)
split <- sample.split(data$label, SplitRatio = 0.80)
train_bow <- dtm_bow[split, ]
test_bow  <- dtm_bow[!split, ]

y_train <- factor(data$label[split])
y_test  <- factor(data$label[!split])

convert_counts <- function(x) {
  ifelse(x > 0, "Yes", "No")
}

train_nb_data <- apply(as.matrix(train_bow), MARGIN = 2, convert_counts)
test_nb_data  <- apply(as.matrix(test_bow),  MARGIN = 2, convert_counts)

nb_model <- naiveBayes(x = train_nb_data, y = y_train, laplace = 1)

nb_predictions <- predict(nb_model, newdata = test_nb_data)

nb_cm <- table(Predicted = nb_predictions, Actual = y_test)

ggplot(as.data.frame(nb_cm), aes(x = Actual, y = Predicted, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), size = 6) +
  scale_fill_gradient(low = "red", high = "green") +
  labs(
    title = "Naive Bayes (BoW) Confusion Matrix", 
    x = "Actual Class", 
    y = "Predicted Class", 
    fill = "Count"
  ) +
  theme_minimal()

x_train_lr <- as.matrix(train_bow)
x_test_lr  <- as.matrix(test_bow)

lr_model <- cv.glmnet(x = x_train_lr, y = y_train, family = "binomial")

lr_pred_raw <- predict(lr_model, newx = x_test_lr, type = "class")

lr_predictions <- factor(lr_pred_raw, levels = levels(y_test))

nb_TP <- nb_cm["spam", "spam"]
nb_TN <- nb_cm["ham", "ham"]
nb_FP <- nb_cm["spam", "ham"]
nb_FN <- nb_cm["ham", "spam"]

nb_accuracy <- (nb_TP + nb_TN) / (nb_TP + nb_TN + nb_FP + nb_FN)
nb_precision <- nb_TP / (nb_TP + nb_FP)
nb_recall <- nb_TP / (nb_TP + nb_FN)
nb_f1 <- 2 * (nb_precision * nb_recall) / (nb_precision + nb_recall)

lr_cm <- table(Predicted = lr_predictions, Actual = y_test)

ggplot(as.data.frame(lr_cm), aes(x = Actual, y = Predicted, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), size = 6) +
  scale_fill_gradient(low = "lightblue", high = "blue") + 
  labs(
    title = "Logistic Regression (BoW) Confusion Matrix", 
    x = "Actual Class", 
    y = "Predicted Class", 
    fill = "Count"
  ) +
  theme_minimal()

lr_TP <- lr_cm["spam", "spam"]
lr_TN <- lr_cm["ham", "ham"]
lr_FP <- lr_cm["spam", "ham"]
lr_FN <- lr_cm["ham", "spam"]

lr_accuracy <- (lr_TP + lr_TN) / (lr_TP + lr_TN + lr_FP + lr_FN)
lr_precision <- lr_TP / (lr_TP + lr_FP)
lr_recall <- lr_TP / (lr_TP + lr_FN)
lr_f1 <- 2 * (lr_precision * lr_recall) / (lr_precision + lr_recall)


train_corpus <- corpus_preprocess[split]
test_corpus <- corpus_preprocess[!split]

train_tfidf <- DocumentTermMatrix(
  train_corpus,
  control = list(weighting = weightTfIdf)
)
test_tfidf <- DocumentTermMatrix(
  test_corpus,
  control = list(
    dictionary = Terms(train_tfidf),
    weighting = weightTfIdf
  )
)

x_train_tfidf <- as.matrix(train_tfidf)
x_test_tfidf  <- as.matrix(test_tfidf)

nb_tfidf_model <- naiveBayes(x = x_train_tfidf, y = y_train)
nb_tfidf_pred <- predict(nb_tfidf_model, newdata = x_test_tfidf)

lr_tfidf_model <- cv.glmnet(x = x_train_tfidf, y = y_train, family = "binomial")
lr_tfidf_pred_raw <- predict(lr_tfidf_model, newx = x_test_tfidf, type = "class")
lr_tfidf_pred <- factor(as.vector(lr_tfidf_pred_raw), levels = levels(y_test))


nb_tfidf_cm <- table(Predicted = nb_tfidf_pred, Actual = y_test)

ggplot(as.data.frame(nb_tfidf_cm), aes(x = Actual, y = Predicted, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), size = 6) +
  scale_fill_gradient(low = "red", high = "green") +
  labs(title = "Naive Bayes (TF-IDF) Confusion Matrix", x = "Actual Class", y = "Predicted Class", fill = "Count") +
  theme_minimal()

nb_tfidf_TP <- nb_tfidf_cm["spam", "spam"]
nb_tfidf_TN <- nb_tfidf_cm["ham", "ham"]
nb_tfidf_FP <- nb_tfidf_cm["spam", "ham"]
nb_tfidf_FN <- nb_tfidf_cm["ham", "spam"]

nb_tfidf_accuracy <- (nb_tfidf_TP + nb_tfidf_TN) / sum(nb_tfidf_cm)
nb_tfidf_precision <- nb_tfidf_TP / (nb_tfidf_TP + nb_tfidf_FP)
nb_tfidf_recall <- nb_tfidf_TP / (nb_tfidf_TP + nb_tfidf_FN)
nb_tfidf_f1 <- 2 * nb_tfidf_precision * nb_tfidf_recall / (nb_tfidf_precision + nb_tfidf_recall)


lr_tfidf_cm <- table(Predicted = lr_tfidf_pred, Actual = y_test)

ggplot(as.data.frame(lr_tfidf_cm), aes(x = Actual, y = Predicted, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), size = 6) +
  labs(title = "Logistic Regression (TF-IDF) Confusion Matrix", x = "Actual Class", y = "Predicted Class", fill = "Count") +
  theme_minimal()

lr_tfidf_TP <- lr_tfidf_cm["spam", "spam"]
lr_tfidf_TN <- lr_tfidf_cm["ham", "ham"]
lr_tfidf_FP <- lr_tfidf_cm["spam", "ham"]
lr_tfidf_FN <- lr_tfidf_cm["ham", "spam"]

lr_tfidf_accuracy <- (lr_tfidf_TP + lr_tfidf_TN) / sum(lr_tfidf_cm)
lr_tfidf_precision <- lr_tfidf_TP / (lr_tfidf_TP + lr_tfidf_FP)
lr_tfidf_recall <- lr_tfidf_TP / (lr_tfidf_TP + lr_tfidf_FN)
lr_tfidf_f1 <- 2 * (lr_tfidf_precision * lr_tfidf_recall) / (lr_tfidf_precision + lr_tfidf_recall)

test_text_raw <- data$text[!split]

generate_noisy_text <- function(text){
  words <- unlist(strsplit(text, " "))
  if(length(words) > 3){
    delete_index <- sample(1:length(words), 1)
    words <- words[-delete_index]
    
    if (length(words) > 0) {
      typo_index <- sample(1:length(words), 1)
      if(nchar(words[typo_index]) > 3){
        chars <- strsplit(words[typo_index], "")[[1]]
        swap_pos <- sample(1:(length(chars)-1), 1)
        temp <- chars[swap_pos]
        chars[swap_pos] <- chars[swap_pos+1]
        chars[swap_pos+1] <- temp
        words[typo_index] <- paste(chars, collapse = "")
      }
    }
  }
  return(paste(words, collapse = " "))
}

set.seed(123)
noisy_test_text <- sapply(test_text_raw, generate_noisy_text)

noisy_corpus <- VCorpus(VectorSource(noisy_test_text))
noisy_corpus <- tm_map(noisy_corpus, content_transformer(tolower))
noisy_corpus <- tm_map(noisy_corpus, removeWords, stopwords("en"))
noisy_corpus <- tm_map(noisy_corpus, stemDocument)
noisy_corpus <- tm_map(noisy_corpus, stripWhitespace)

noisy_test_bow <- DocumentTermMatrix(
  noisy_corpus, 
  control = list(dictionary = Terms(train_bow))
)

noisy_test_nb_data <- apply(as.matrix(noisy_test_bow), MARGIN = 2, convert_counts)
nb_noisy_pred <- predict(nb_model, newdata = noisy_test_nb_data)

nb_noisy_cm <- table(Predicted = nb_noisy_pred, Actual = y_test)

nb_n_TP <- nb_noisy_cm["spam", "spam"]
nb_n_TN <- nb_noisy_cm["ham", "ham"]
nb_n_FP <- nb_noisy_cm["spam", "ham"]
nb_n_FN <- nb_noisy_cm["ham", "spam"]

nb_noisy_accuracy <- (nb_n_TP + nb_n_TN) / (nb_n_TP + nb_n_TN + nb_n_FP + nb_n_FN)
nb_noisy_precision <- nb_n_TP / (nb_n_TP + nb_n_FP)
nb_noisy_recall <- nb_n_TP / (nb_n_TP + nb_n_FN)
nb_noisy_f1 <- 2 * (nb_noisy_precision * nb_noisy_recall) / (nb_noisy_precision + nb_noisy_recall)


x_noisy_test_lr <- as.matrix(noisy_test_bow)
lr_noisy_pred_raw <- predict(lr_model, newx = x_noisy_test_lr, type = "class")
lr_noisy_pred <- factor(as.vector(lr_noisy_pred_raw), levels = levels(y_test))

lr_noisy_cm <- table(Predicted = lr_noisy_pred, Actual = y_test)

lr_n_TP <- lr_noisy_cm["spam", "spam"]
lr_n_TN <- lr_noisy_cm["ham", "ham"]
lr_n_FP <- lr_noisy_cm["spam", "ham"]
lr_n_FN <- lr_noisy_cm["ham", "spam"]

lr_noisy_accuracy <- (lr_n_TP + lr_n_TN) / (lr_n_TP + lr_n_TN + lr_n_FP + lr_n_FN)
lr_noisy_precision <- lr_n_TP / (lr_n_TP + lr_n_FP)
lr_noisy_recall <- lr_n_TP / (lr_n_TP + lr_n_FN)
lr_noisy_f1 <- 2 * (lr_noisy_precision * lr_noisy_recall) / (lr_noisy_precision + lr_noisy_recall)

noisy_test_tfidf <- DocumentTermMatrix(
  noisy_corpus,
  control = list(
    dictionary = Terms(train_tfidf),
    weighting = weightTfIdf
  )
)

x_noisy_test_tfidf <- as.matrix(noisy_test_tfidf)

nb_tfidf_noisy_pred <- predict(nb_tfidf_model, newdata = x_noisy_test_tfidf)

nb_tfidf_noisy_cm <- table(Predicted = nb_tfidf_noisy_pred, Actual = y_test)

nb_tf_n_TP <- nb_tfidf_noisy_cm["spam", "spam"]
nb_tf_n_TN <- nb_tfidf_noisy_cm["ham", "ham"]
nb_tf_n_FP <- nb_tfidf_noisy_cm["spam", "ham"]
nb_tf_n_FN <- nb_tfidf_noisy_cm["ham", "spam"]

nb_tfidf_noisy_accuracy <- (nb_tf_n_TP + nb_tf_n_TN) / (nb_tf_n_TP + nb_tf_n_TN + nb_tf_n_FP + nb_tf_n_FN)
nb_tfidf_noisy_precision <- nb_tf_n_TP / (nb_tf_n_TP + nb_tf_n_FP)
nb_tfidf_noisy_recall <- nb_tf_n_TP / (nb_tf_n_TP + nb_tf_n_FN)
nb_tfidf_noisy_f1 <- 2 * (nb_tfidf_noisy_precision * nb_tfidf_noisy_recall) / (nb_tfidf_noisy_precision + nb_tfidf_noisy_recall)

lr_tfidf_noisy_pred_raw <- predict(lr_tfidf_model, newx = x_noisy_test_tfidf, type = "class")
lr_tfidf_noisy_pred <- factor(as.vector(lr_tfidf_noisy_pred_raw), levels = levels(y_test))

lr_tfidf_noisy_cm <- table(Predicted = lr_tfidf_noisy_pred, Actual = y_test)

lr_tf_n_TP <- lr_tfidf_noisy_cm["spam", "spam"]
lr_tf_n_TN <- lr_tfidf_noisy_cm["ham", "ham"]
lr_tf_n_FP <- lr_tfidf_noisy_cm["spam", "ham"]
lr_tf_n_FN <- lr_tfidf_noisy_cm["ham", "spam"]

lr_tfidf_noisy_accuracy <- (lr_tf_n_TP + lr_tf_n_TN) / (lr_tf_n_TP + lr_tf_n_TN + lr_tf_n_FP + lr_tf_n_FN)
lr_tfidf_noisy_precision <- lr_tf_n_TP / (lr_tf_n_TP + lr_tf_n_FP)
lr_tfidf_noisy_recall <- lr_tf_n_TP / (lr_tf_n_TP + lr_tf_n_FN)
lr_tfidf_noisy_f1 <- 2 * (lr_tfidf_noisy_precision * lr_tfidf_noisy_recall) / (lr_tfidf_noisy_precision + lr_tfidf_noisy_recall)

results_df <- data.frame(
  Model = c("Naive Bayes (BoW)", "Naive Bayes (BoW)",
            "Logistic Reg (BoW)", "Logistic Reg (BoW)",
            "Naive Bayes (TF-IDF)", "Naive Bayes (TF-IDF)",
            "Logistic Reg (TF-IDF)", "Logistic Reg (TF-IDF)"),
  
  Condition = factor(c("Clean", "Noisy", "Clean", "Noisy", "Clean", "Noisy", "Clean", "Noisy"), 
                     levels = c("Clean", "Noisy")),
  
  Accuracy = c(nb_accuracy, nb_noisy_accuracy,
               lr_accuracy, lr_noisy_accuracy,
               nb_tfidf_accuracy, nb_tfidf_noisy_accuracy,
               lr_tfidf_accuracy, lr_tfidf_noisy_accuracy),
  
  F1_Score = c(nb_f1, nb_noisy_f1,
               lr_f1, lr_noisy_f1,
               nb_tfidf_f1, nb_tfidf_noisy_f1,
               lr_tfidf_f1, lr_tfidf_noisy_f1)
)

results_df


comparison_plot <- ggplot(results_df, aes(x = Model, y = F1_Score, fill = Condition)) +
  geom_bar(stat = "identity", position = "dodge", color = "black", alpha = 0.8) +
  scale_fill_manual(values = c("Clean" = "#4CAF50", "Noisy" = "#F44336")) +
  labs(
    title = "Clean vs Noisy Performance Comparison (F1 Score)",
    subtitle = "Assessing model robustness against spelling errors and character deletion",
    x = "Model & Feature Extraction Method",
    y = "F1 Score",
    fill = "Text Condition"
  ) +
  coord_cartesian(ylim = c(0, 1)) + 
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 10, face = "bold"))
comparison_plot

feature_comparison_df <- data.frame(
  Model = c("Naive Bayes", "Naive Bayes", 
            "Logistic Regression", "Logistic Regression"),
  
  Feature_Extraction = factor(c("BoW", "TF-IDF", "BoW", "TF-IDF"), 
                              levels = c("BoW", "TF-IDF")),
  
  Accuracy = c(nb_accuracy, nb_tfidf_accuracy, 
               lr_accuracy, lr_tfidf_accuracy),
  
  F1_Score = c(nb_f1, nb_tfidf_f1, 
               lr_f1, lr_tfidf_f1)
)

feature_plot <- ggplot(feature_comparison_df, aes(x = Model, y = F1_Score, fill = Feature_Extraction)) +
  geom_bar(stat = "identity", position = "dodge", color = "black", alpha = 0.85) +
  scale_fill_manual(values = c("BoW" = "#4CAF50", "TF-IDF" = "#F44336")) +
  labs(
    title = "Comparison: BoW vs TF-IDF",
    subtitle = "Evaluating baseline performance on clean text data",
    x = "Model",
    y = "F1 Score",
    fill = "Extraction Method"
  ) +
  coord_cartesian(ylim = c(0, 1)) + 
  theme_minimal() +
  theme(
    axis.text.x = element_text(size = 12, face = "bold"),
    plot.title = element_text(face = "bold", size = 14)
  )
feature_plot
