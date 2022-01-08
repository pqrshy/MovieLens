

# Project Movielens
# Author: Poonam Quraishy
# Date: December 2021




# Importing data from code provided within the course.


##########################################################
# Create edx set, validation set 
##########################################################

# Note: This script could run for about 10-15 minutes,
# run-time may be sooner or later depending on system capability.

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library(lubridate)
library(ggplot2)
library(knitr)
library(kableExtra)
library(tinytex)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")


movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)


###################################################################
# Exploring and analyzing the dataset.
###################################################################


head(edx)
str(edx)
edx %>% select(-genres) %>% summary()
dim(edx)

# check for NA value 
anyNA(edx)

# Number of unique movies and users in the edx dataset 
edx %>%
  summarize(un_users = n_distinct(userId),
            un_genres = n_distinct(genres),
            un_movies = n_distinct(movieId))

# Plot of Ratings
edx %>%
  ggplot(aes(rating)) +
  geom_histogram(binwidth = 0.5, color = "black", fill = "steelblue") +
  xlab("Rating") +
  ylab("Count") +
  ggtitle("Ratings") +
  theme(plot.title = element_text(hjust = 0.5))


# The most rated movies in descending order.

edx %>% group_by(movieId, title) %>%
  summarize(count = n()) %>%
  arrange(desc(count))

# Five most given ratings from most to least.

edx %>% group_by(rating) %>% summarize(count = n()) %>% top_n(5) %>%
  arrange(desc(count))

# Visualizing user ratings by grouping movies by genre.

moviesbygenre <- edx %>% separate_rows(genres, sep = "\\|") %>%
  group_by(genres) %>%
  summarize(count = n()) %>%
  arrange(desc(count))

moviesbygenre <- data.table(moviesbygenre)
moviesbygenre <- moviesbygenre[order(-count),]
ggplot(data=moviesbygenre, aes(x=reorder(moviesbygenre$genres,moviesbygenre$count),
                               y=sapply(moviesbygenre$count, function(y) y/1000000),
                               fill=I("steelblue"))) +
  geom_bar(position="dodge",stat="identity") + 
  coord_flip() +
  labs(x="Movie Genre", y="Number of User Ratings in Millions", 
       caption = "source: Edx Movielens dataset") +
  ggtitle("User Ratings by Movie Genre")


# Number of ratings per movie
edx %>%
  count(movieId) %>%
  ggplot(aes(n)) +
  geom_histogram(bins = 30, color = "black", fill= "steelblue") +
  scale_x_log10() +
  labs(x = "Number of ratings",
       caption = "source: Edx Movielens dataset") +
  ylab("Number of movies") +
  ggtitle("Number of ratings per movie") +
  theme_gray()

# Number of rating per user

edx %>% count(userId) %>% ggplot(aes(n))+
  geom_histogram(binwidth = 0.05, color = "black" , fill= "steelblue")+
  labs(x = "Ratings per user",
       caption = "source: Edx Movielens dataset") +
  ylab("Number of users") +
  ggtitle("Number of Ratings by Users")+
  scale_x_log10()+
  theme_gray()

# Mean user Ratings.

edx %>%
  group_by(userId) %>%
  filter(n() >= 100) %>%
  summarize(m_u = mean(rating)) %>%
  ggplot(aes(m_u)) +
  geom_histogram(bins = 30, color = "black", fill= "steelblue") +
  labs(x = "Mean Ratings",
       caption = "source: Edx Movielens dataset") +
  ylab("Number of users") +
  ggtitle("Mean user movie ratings") +
  scale_x_continuous(breaks =  c(seq(0.5,5,0.5))) +
  theme_gray()



###################################################################
# Exploratory Data Analysis & Modeling Approach           
##################################################################

#Split the Edx dataset into train & test set.
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.2, list = FALSE)
train_set <- edx[-test_index,]
test_set <- edx[test_index,]


# Removing entries (users/movies) present in test set to ensure that they don't appear in train set, we will use semi_join()
test_set <- test_set %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")



# Computing the loss function

RMSE <- function(actual_ratings, predicted_ratings) {
  sqrt(mean((actual_ratings - predicted_ratings)^2))}


# Model - 1 

# Taking the average of all observed ratings.
mu <- mean(train_set$rating)
mu

# Test results based on simple prediction
meanmodel_rmse <- RMSE(test_set$rating, mu)
meanmodel_rmse

# The results of the mean rating model are displayed 

rmse_results <- data_frame(Method = "Mean Rating Model",Dataset= "Edx_Test", RMSE = meanmodel_rmse)
rmse_results %>% knitr::kable()

# Model - 2
# Model 2 with added movie effect

movie_averages <- train_set %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu))

movie_averages %>% qplot(b_i, geom ="histogram", bins = 10, data = ., color = I("steelblue"),
                         ylab = "Number of movies", main = "Number of movies with the movie effect")

b_i <- test_set %>% 
  left_join(movie_averages, by='movieId') %>%
  .$b_i

# Test and save rmse results 
predicted_ratings <- mu + b_i
movieeffect_rmse <- RMSE(predicted_ratings, test_set$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(Method="Movie effect model",Dataset="Edx_Test",   
                                     RMSE = movieeffect_rmse ))

# Check results
rmse_results %>% knitr::kable()

# Model 3 with movie and user effects

user_averages <- train_set %>%
  left_join(movie_averages, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

b_u <- test_set %>% 
  left_join(movie_averages, by='movieId') %>%
  left_join(user_averages, by='userId') %>%
  .$b_u

predicted_ratings <- mu + b_i + b_u

# check and save results

usereffect_rmse <- RMSE(predicted_ratings, test_set$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(Method="Movie and user effect model",Dataset="Edx_Test",
                                     RMSE = usereffect_rmse))
rmse_results %>% knitr::kable()

# Model 4 with Regularization

lambdas <- seq(0, 5, 0.50)
rmses <- sapply(lambdas, function(l){
  mu <- mean(train_set$rating)
  
  movie_averages <- train_set %>%
    group_by(movieId) %>%
    summarize(movie_averages = sum(rating - mu)/(n()+l))
  
  user_averages <- train_set %>%
    left_join(movie_averages, by='movieId') %>%
    group_by(userId) %>%
    summarize(user_averages = sum(rating - mu - movie_averages)/(n()+l))
  
  predicted_ratings <- test_set %>% 
    left_join(movie_averages, by = "movieId") %>%
    left_join(user_averages, by = "userId") %>%
    mutate(pred = mu + movie_averages + user_averages) %>%
    pull(pred)
  
  return(RMSE(predicted_ratings, test_set$rating))
})

# Optimal Lambda

qplot(lambdas, rmses)

lambda <- lambdas[which.min(rmses)]
lambda

rmse_results <- bind_rows(rmse_results,
                          data_frame(Method="Model with Regularization",Dataset ="Edx_Test",  
                                     RMSE = min(rmses)))
rmse_results %>% knitr::kable()

# Using the model with Regularization on the hold-out Validation test data set.

mu <- mean(validation$rating)
l <- 5
b_i <- validation %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n() + l))

b_u <- validation %>%
  left_join(b_i, by='movieId') %>% 
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n() +l))

predicted_ratings <- validation %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  mutate(pred = mu + b_i +  b_u) %>% .$pred

RMSE(predicted_ratings, validation$rating)

model_regularization <- RMSE(predicted_ratings, validation$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(Method = "Model with Regularization", Dataset="Validation",  
                                     RMSE = model_regularization))

rmse_results %>% knitr::kable()



#### Appendix ####
print("Operating System:")
version





