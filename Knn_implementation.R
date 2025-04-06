# implementation of knn algorithm in classification

#install packages
library(tidyverse)
library(ggplot2)

df <- iris
df

#visualizing the data

ggplot(data = data, mapping = aes(x = Sepal.Length/Sepal.Width, y = Petal.Length/Petal.Width, 
                                  colour = Species, shape = Species)) +
  geom_point() + labs(title = "Iris Data",
                      x = "Sepal Proportion",
                      y = "Petal Proportion") + 
  theme(plot.title = element_text(hjust = 0.5))

#create test and train data
set.seed(42L) # to ensure reproducibility

#splitting our data proportionally into train data at 4:1 ratio
train_index <- sample(1:nrow(iris), size = 0.8 * nrow(iris))

# defining the test data: which everything not sampled in trainig
test_index <- setdiff(1:nrow(iris), train_index)

#create the test and train datasets
x_train <- df[train_index, -ncol(iris)]
y_train <- df[train_index, "Species"]

x_test <- df[test_index, -5]
y_test <- iris[test_index, 5]

head(x_train)
dim(x_train)
dim(x_test)
length(y_train)

#function to calculate Euclidean distance of two observations
dist <- function(p, q){
  (p-q)**2 %>% sum %>% sqrt
}
# below also works
#dist <- sqrt(sum((p-q)**2))

#taking the first observation of test and calculating its distance to all observations in the training
#we know we are only dealing with x_test and x_train as we are focused on observations

v_dist <- vector(mode = "numeric", length = 120L)

for (i in 1:120){
  v_dist[i] = dist(x_test[1,], x_train[i,])
}
v_dist %>% head()

#finding the species of the three nearest neighbours to the first observation 
library(kit)
nn = topn(v_dist, n = 3, decreasing = FALSE)
nn_spec = y_train[nn]
nn_spec

#predict the species of the first test observation
sort_tab = table(nn_spec) %>% sort(decreasing = TRUE)
prediction = names(sort_tab[1])
prediction == y_test[1]

#create a function to get the nearest neighbours
knn = function(x, x_train, y_train, k) {
  v_dist = apply(x_train, 1, dist, q = x)
  nn = topn(v_dist, n = k, decreasing = FALSE)
  nn_spec = y_train[nn]
  sort_tab = table(nn_spec) %>% sort(decreasing = TRUE)
  names(sort_tab[1])
}

#apply the knn function across all test data
sol = apply(X = x_test,
            MARGIN = 1,
            FUN = knn,
            x_train = x_train,
            y_train = y_train,
            k = 5)
sol %>% head

#print confusion matrix
table(sol, y_test)



