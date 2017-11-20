
# coding: utf-8

# # Machine Learning Engineer Nanodegree
# ## Unsupervised Learning
# ## Project: Creating Customer Segments

# Welcome to the third project of the Machine Learning Engineer Nanodegree! In this notebook, some template code has already been provided for you, and it will be your job to implement the additional functionality necessary to successfully complete this project. Sections that begin with **'Implementation'** in the header indicate that the following block of code will require additional functionality which you must provide. Instructions will be provided for each section and the specifics of the implementation are marked in the code block with a `'TODO'` statement. Please be sure to read the instructions carefully!
# 
# In addition to implementing code, there will be questions that you must answer which relate to the project and your implementation. Each section where you will answer a question is preceded by a **'Question X'** header. Carefully read each question and provide thorough answers in the following text boxes that begin with **'Answer:'**. Your project submission will be evaluated based on your answers to each of the questions and the implementation you provide.  
# 
# >**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut. In addition, Markdown cells can be edited by typically double-clicking the cell to enter edit mode.

# ## Getting Started
# 
# In this project, you will analyze a dataset containing data on various customers' annual spending amounts (reported in *monetary units*) of diverse product categories for internal structure. One goal of this project is to best describe the variation in the different types of customers that a wholesale distributor interacts with. Doing so would equip the distributor with insight into how to best structure their delivery service to meet the needs of each customer.
# 
# The dataset for this project can be found on the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Wholesale+customers). For the purposes of this project, the features `'Channel'` and `'Region'` will be excluded in the analysis — with focus instead on the six product categories recorded for customers.
# 
# Run the code block below to load the wholesale customers dataset, along with a few of the necessary Python libraries required for this project. You will know the dataset loaded successfully if the size of the dataset is reported.

# In[1]:

# Import libraries necessary for this project
import numpy as np
import pandas as pd
from IPython.display import display # Allows the use of display() for DataFrames

# Import supplementary visualizations code visuals.py
import visuals as vs

# Pretty display for notebooks
get_ipython().magic(u'matplotlib inline')

# Load the wholesale customers dataset
try:
    data = pd.read_csv("customers.csv")
    data.drop(['Region', 'Channel'], axis = 1, inplace = True)
    print "Wholesale customers dataset has {} samples with {} features each.".format(*data.shape)
except:
    print "Dataset could not be loaded. Is the dataset missing?"


# ## Data Exploration
# In this section, you will begin exploring the data through visualizations and code to understand how each feature is related to the others. You will observe a statistical description of the dataset, consider the relevance of each feature, and select a few sample data points from the dataset which you will track through the course of this project.
# 
# Run the code block below to observe a statistical description of the dataset. Note that the dataset is composed of six important product categories: **'Fresh'**, **'Milk'**, **'Grocery'**, **'Frozen'**, **'Detergents_Paper'**, and **'Delicatessen'**. Consider what each category represents in terms of products you could purchase.

# In[2]:

# Display a description of the dataset
display(data.describe())


# ### Implementation: Selecting Samples
# To get a better understanding of the customers and how their data will transform through the analysis, it would be best to select a few sample data points and explore them in more detail. In the code block below, add **three** indices of your choice to the `indices` list which will represent the customers to track. It is suggested to try different sets of samples until you obtain customers that vary significantly from one another.

# In[3]:

# TODO: Select three indices of your choice you wish to sample from the dataset
indices = [100, 200, 400]#[1,5,10,100, 200, 300, 400]#[10,100,300]

# Create a DataFrame of the chosen samples
samples = pd.DataFrame(data.loc[indices], columns = data.keys()).reset_index(drop = True)
print "Chosen samples of wholesale customers dataset:"
display(samples)


# ### Question 1
# Consider the total purchase cost of each product category and the statistical description of the dataset above for your sample customers.  
# *What kind of establishment (customer) could each of the three samples you've chosen represent?*  
# **Hint:** Examples of establishments include places like markets, cafes, and retailers, among many others. Avoid using names for establishments, such as saying *"McDonalds"* when describing a sample customer as a restaurant.

# **Answer:**  
# * First establishment: The fresh food and grocery is higher comparing to other categories. It may be a small resturant.
# * Seconde establishment: The grocety is much higher than others. It might be a samll grocery store.
# * Third establishment: The spends of each categories is low, and the fresh is a little bit higher than others. It might be a small food truck.

# ### Implementation: Feature Relevance
# One interesting thought to consider is if one (or more) of the six product categories is actually relevant for understanding customer purchasing. That is to say, is it possible to determine whether customers purchasing some amount of one category of products will necessarily purchase some proportional amount of another category of products? We can make this determination quite easily by training a supervised regression learner on a subset of the data with one feature removed, and then score how well that model can predict the removed feature.
# 
# In the code block below, you will need to implement the following:
#  - Assign `new_data` a copy of the data by removing a feature of your choice using the `DataFrame.drop` function.
#  - Use `sklearn.cross_validation.train_test_split` to split the dataset into training and testing sets.
#    - Use the removed feature as your target label. Set a `test_size` of `0.25` and set a `random_state`.
#  - Import a decision tree regressor, set a `random_state`, and fit the learner to the training data.
#  - Report the prediction score of the testing set using the regressor's `score` function.

# In[4]:

# TODO: Make a copy of the DataFrame, using the 'drop' function to drop the given feature
new_data = pd.DataFrame(data=data.drop(['Grocery'], axis=1))

# # TODO: Split the data into training and testing sets using the given feature as the target
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(new_data, data['Grocery'], test_size=0.25, random_state=70)

# # TODO: Create a decision tree regressor and fit it to the training set
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X_train, y_train)

# # TODO: Report the score of the prediction using the testing set
score = regressor.score(X_test, y_test)
print score
#display(new_data.describe())


# ### Question 2
# *Which feature did you attempt to predict? What was the reported prediction score? Is this feature necessary for identifying customers' spending habits?*  
# **Hint:** The coefficient of determination, `R^2`, is scored between 0 and 1, with 1 being a perfect fit. A negative `R^2` implies the model fails to fit the data.

# **Answer:**  
# The Grocery feature is chosen, and the score is 0.7575. According to the score, this feature can be considered as a customer's spending habits.

# ### Visualize Feature Distributions
# To get a better understanding of the dataset, we can construct a scatter matrix of each of the six product features present in the data. If you found that the feature you attempted to predict above is relevant for identifying a specific customer, then the scatter matrix below may not show any correlation between that feature and the others. Conversely, if you believe that feature is not relevant for identifying a specific customer, the scatter matrix might show a correlation between that feature and another feature in the data. Run the code block below to produce a scatter matrix.

# In[5]:

# Produce a scatter matrix for each pair of features in the data
pd.scatter_matrix(data, alpha = 0.3, figsize = (14,8), diagonal = 'kde');


# ### Question 3
# *Are there any pairs of features which exhibit some degree of correlation? Does this confirm or deny your suspicions about the relevance of the feature you attempted to predict? How is the data for those features distributed?*  
# **Hint:** Is the data normally distributed? Where do most of the data points lie? 

# **Answer:**  
# * Grocery and Detetgents_Paper have positive correlation according to the scatter plot.  
# * The predict feature is Grocery. When we use other features to predict the result, it seems the result will be affected by some features which do not show correlation. Therefore, the score is not really high.  
# * Comparing Grocery to Fresh, Delicatessen and Frozen, it shows ngative correlation between them. Comparing Grocery to Milk and Detergents_Paper, it shows posistive correlation between.  
# * The data looks not normally distributed. Most point is under 10k.

# ## Data Preprocessing
# In this section, you will preprocess the data to create a better representation of customers by performing a scaling on the data and detecting (and optionally removing) outliers. Preprocessing data is often times a critical step in assuring that results you obtain from your analysis are significant and meaningful.

# ### Implementation: Feature Scaling
# If data is not normally distributed, especially if the mean and median vary significantly (indicating a large skew), it is most [often appropriate](http://econbrowser.com/archives/2014/02/use-of-logarithms-in-economics) to apply a non-linear scaling — particularly for financial data. One way to achieve this scaling is by using a [Box-Cox test](http://scipy.github.io/devdocs/generated/scipy.stats.boxcox.html), which calculates the best power transformation of the data that reduces skewness. A simpler approach which can work in most cases would be applying the natural logarithm.
# 
# In the code block below, you will need to implement the following:
#  - Assign a copy of the data to `log_data` after applying logarithmic scaling. Use the `np.log` function for this.
#  - Assign a copy of the sample data to `log_samples` after applying logarithmic scaling. Again, use `np.log`.

# In[6]:

# TODO: Scale the data using the natural logarithm
log_data = np.log(data)

# TODO: Scale the sample data using the natural logarithm
log_samples = np.log(log_data)

# Produce a scatter matrix for each pair of newly-transformed features
pd.scatter_matrix(log_data, alpha = 0.3, figsize = (14,8), diagonal = 'kde');


# ### Observation
# After applying a natural logarithm scaling to the data, the distribution of each feature should appear much more normal. For any pairs of features you may have identified earlier as being correlated, observe here whether that correlation is still present (and whether it is now stronger or weaker than before).
# 
# Run the code below to see how the sample data has changed after having the natural logarithm applied to it.

# In[7]:

# Display the log-transformed sample data
display(log_samples.head(n=5))


# ### Implementation: Outlier Detection
# Detecting outliers in the data is extremely important in the data preprocessing step of any analysis. The presence of outliers can often skew results which take into consideration these data points. There are many "rules of thumb" for what constitutes an outlier in a dataset. Here, we will use [Tukey's Method for identfying outliers](http://datapigtechnologies.com/blog/index.php/highlighting-outliers-in-your-data-with-the-tukey-method/): An *outlier step* is calculated as 1.5 times the interquartile range (IQR). A data point with a feature that is beyond an outlier step outside of the IQR for that feature is considered abnormal.
# 
# In the code block below, you will need to implement the following:
#  - Assign the value of the 25th percentile for the given feature to `Q1`. Use `np.percentile` for this.
#  - Assign the value of the 75th percentile for the given feature to `Q3`. Again, use `np.percentile`.
#  - Assign the calculation of an outlier step for the given feature to `step`.
#  - Optionally remove data points from the dataset by adding indices to the `outliers` list.
# 
# **NOTE:** If you choose to remove any outliers, ensure that the sample data does not contain any of these points!  
# Once you have performed this implementation, the dataset will be stored in the variable `good_data`.

# In[14]:

# For each feature find the data points with extreme high or low values
list_outlier = []
list_outlier_conditions = []
for feature in log_data.keys():
    # TODO: Calculate Q1 (25th percentile of the data) for the given feature
    Q1 = np.percentile(a=log_data[feature],q=25)
    
    # TODO: Calculate Q3 (75th percentile of the data) for the given feature
    Q3 = np.percentile(a=log_data[feature],q=75)
    
    # TODO: Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)
    step = (Q3-Q1)*1.5
    
    # Display the outliers
    print "Data points considered outliers for the feature '{}':".format(feature)
    #print "Q1 - step: %f, Q3 + step: %f" %(Q1 - step, Q3 + step)
    outlier_data = log_data[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))]
    display(outlier_data)
    # record outlier's minimum and maximum values of this feature
    list_outlier_conditions.append("%s: Q1 - step: %f, Q3 + step: %f" %(feature, Q1 - step, Q3 + step))
    # record outlier's indices
    list_outlier.append(outlier_data.index.values)
    #union_outlier = pd.merge(union_outlier, feature_data, how='right', on=list(feature_data.columns.values))
#display(union_outlier)
# OPTIONAL: Select the indices for data points you wish to remove
outliers  = [65, 66, 75, 128, 154]#[]

# Remove the outliers, if any were specified
good_data = log_data.drop(log_data.index[outliers]).reset_index(drop = True)


# In[13]:

# Get the data points wich contans more than two outlier features
features_name = list(log_data.columns.values)
for i in range(0, len(list_outlier)):
    for j in range(i, len(list_outlier)):
        if i != j:
            intersection = np.intersect1d(list_outlier[i], list_outlier[j])
            if intersection.size > 0:
                print "%s, %s => %s" %(features_name[i], features_name[j], intersection)
#display(list_outlier)
display(list_outlier_conditions)


# ### Question 4
# *Are there any data points considered outliers for more than one feature based on the definition above? Should these data points be removed from the dataset? If any data points were added to the `outliers` list to be removed, explain why.* 

# **Answer:**  
# * There are five data points containg more than two outlier features. They are 65, 66, 75, 128, and 154.
# * Yes, I would like to remove the ourliers because they affect the training a lot.
# * I will remove all data points which contains more than one outlier features. Removing these outliers can prevent the result distorted.

# ## Feature Transformation
# In this section you will use principal component analysis (PCA) to draw conclusions about the underlying structure of the wholesale customer data. Since using PCA on a dataset calculates the dimensions which best maximize variance, we will find which compound combinations of features best describe customers.

# ### Implementation: PCA
# 
# Now that the data has been scaled to a more normal distribution and has had any necessary outliers removed, we can now apply PCA to the `good_data` to discover which dimensions about the data best maximize the variance of features involved. In addition to finding these dimensions, PCA will also report the *explained variance ratio* of each dimension — how much variance within the data is explained by that dimension alone. Note that a component (dimension) from PCA can be considered a new "feature" of the space, however it is a composition of the original features present in the data.
# 
# In the code block below, you will need to implement the following:
#  - Import `sklearn.decomposition.PCA` and assign the results of fitting PCA in six dimensions with `good_data` to `pca`.
#  - Apply a PCA transformation of `log_samples` using `pca.transform`, and assign the results to `pca_samples`.

# In[20]:

# TODO: Apply PCA by fitting the good data with the same number of dimensions as features
from sklearn.decomposition import PCA
pca = PCA()
pca.fit(good_data)

# TODO: Transform log_samples using the PCA fit above
pca_samples = pca.transform(log_samples)

# Generate PCA results plot
pca_results = vs.pca_results(good_data, pca)


# ### Question 5
# *How much variance in the data is explained* ***in total*** *by the first and second principal component? What about the first four principal components? Using the visualization provided above, discuss what the first four dimensions best represent in terms of customer spending.*  
# **Hint:** A positive increase in a specific dimension corresponds with an *increase* of the *positive-weighted* features and a *decrease* of the *negative-weighted* features. The rate of increase or decrease is based on the individual feature weights.

# **Answer:**
# * The first principal component has 0.443 variance, and the second pricipal component has 0.2638 variance.
# * The first four components' variance are 0.443, 0.2638, 0.1231, and 0.1012.  
# The first principal component shows that customers usually purchase Fresh and Frozen product together.  
# The second pricipal component shows that customers who do not by Fresh products usually do not buy Frozen and Delicatessen products.  
# The third pricipal component shows that customers ususally buy Frozen and Delicatessen products when they do not buy Fresh products.  
# The forth pricipal component shows that customers who purchase a lot of Frozen product will buy some Detergents_Paper product.

# ### Observation
# Run the code below to see how the log-transformed sample data has changed after having a PCA transformation applied to it in six dimensions. Observe the numerical value for the first four dimensions of the sample points. Consider if this is consistent with your initial interpretation of the sample points.

# In[21]:

# Display sample log-data after having a PCA transformation applied
display(pd.DataFrame(np.round(pca_samples, 4), columns = pca_results.index.values))


# ### Implementation: Dimensionality Reduction
# When using principal component analysis, one of the main goals is to reduce the dimensionality of the data — in effect, reducing the complexity of the problem. Dimensionality reduction comes at a cost: Fewer dimensions used implies less of the total variance in the data is being explained. Because of this, the *cumulative explained variance ratio* is extremely important for knowing how many dimensions are necessary for the problem. Additionally, if a signifiant amount of variance is explained by only two or three dimensions, the reduced data can be visualized afterwards.
# 
# In the code block below, you will need to implement the following:
#  - Assign the results of fitting PCA in two dimensions with `good_data` to `pca`.
#  - Apply a PCA transformation of `good_data` using `pca.transform`, and assign the results to `reduced_data`.
#  - Apply a PCA transformation of `log_samples` using `pca.transform`, and assign the results to `pca_samples`.

# In[22]:

# TODO: Apply PCA by fitting the good data with only two dimensions
pca = PCA(n_components=2)
pca.fit(good_data)

# TODO: Transform the good data using the PCA fit above
reduced_data = pca.transform(good_data)

# TODO: Transform log_samples using the PCA fit above
pca_samples = pca.transform(log_samples)

# Create a DataFrame for the reduced data
reduced_data = pd.DataFrame(reduced_data, columns = ['Dimension 1', 'Dimension 2'])


# ### Observation
# Run the code below to see how the log-transformed sample data has changed after having a PCA transformation applied to it using only two dimensions. Observe how the values for the first two dimensions remains unchanged when compared to a PCA transformation in six dimensions.

# In[23]:

# Display sample log-data after applying PCA transformation in two dimensions
display(pd.DataFrame(np.round(pca_samples, 4), columns = ['Dimension 1', 'Dimension 2']))


# ## Visualizing a Biplot
# A biplot is a scatterplot where each data point is represented by its scores along the principal components. The axes are the principal components (in this case `Dimension 1` and `Dimension 2`). In addition, the biplot shows the projection of the original features along the components. A biplot can help us interpret the reduced dimensions of the data, and discover relationships between the principal components and original features.
# 
# Run the code cell below to produce a biplot of the reduced-dimension data.

# In[24]:

# Create a biplot
vs.biplot(good_data, reduced_data, pca)


# ### Observation
# 
# Once we have the original feature projections (in red), it is easier to interpret the relative position of each data point in the scatterplot. For instance, a point the lower right corner of the figure will likely correspond to a customer that spends a lot on `'Milk'`, `'Grocery'` and `'Detergents_Paper'`, but not so much on the other product categories. 
# 
# From the biplot, which of the original features are most strongly correlated with the first component? What about those that are associated with the second component? Do these observations agree with the pca_results plot you obtained earlier?

# * The features wich are most strongly correlated with the fist component are Delicatessen, Frozen, and Fresh.
# * The features wich are most strongly correlated with the second component are Detergents_Paper, Grocery, and Milk.
# * Comparing the above plot, they are completely opposite.

# ## Clustering
# 
# In this section, you will choose to use either a K-Means clustering algorithm or a Gaussian Mixture Model clustering algorithm to identify the various customer segments hidden in the data. You will then recover specific data points from the clusters to understand their significance by transforming them back into their original dimension and scale. 

# ### Question 6
# *What are the advantages to using a K-Means clustering algorithm? What are the advantages to using a Gaussian Mixture Model clustering algorithm? Given your observations about the wholesale customer data so far, which of the two algorithms will you use and why?*

# **Answer:**
# * When we provide the number of clusters, the K-Means clustering algorithm can separate samples in groups of equal variance. It is good at a large number of samples and is used to in many different application areas.
# * The Gaussian Mixture Model clustering algorithm uses probabilistic model and the covariance structure of the data to determine clusters. 
# * According to the previous observation, I would like to use the Gaussian Mixture Model clustering algorithm. It is a little bit hard to see clusters on the previous observation. I think the probabilistic model may be more reliable. However, I fialed to implement this algorithm so I will use K-Means.

# ### Implementation: Creating Clusters
# Depending on the problem, the number of clusters that you expect to be in the data may already be known. When the number of clusters is not known *a priori*, there is no guarantee that a given number of clusters best segments the data, since it is unclear what structure exists in the data — if any. However, we can quantify the "goodness" of a clustering by calculating each data point's *silhouette coefficient*. The [silhouette coefficient](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html) for a data point measures how similar it is to its assigned cluster from -1 (dissimilar) to 1 (similar). Calculating the *mean* silhouette coefficient provides for a simple scoring method of a given clustering.
# 
# In the code block below, you will need to implement the following:
#  - Fit a clustering algorithm to the `reduced_data` and assign it to `clusterer`.
#  - Predict the cluster for each data point in `reduced_data` using `clusterer.predict` and assign them to `preds`.
#  - Find the cluster centers using the algorithm's respective attribute and assign them to `centers`.
#  - Predict the cluster for each sample data point in `pca_samples` and assign them `sample_preds`.
#  - Import `sklearn.metrics.silhouette_score` and calculate the silhouette score of `reduced_data` against `preds`.
#    - Assign the silhouette score to `score` and print the result.

# In[40]:

# TODO: Apply your clustering algorithm of choice to the reduced data 
# from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
# clusterer = GaussianMixture()
from sklearn.metrics import silhouette_score
for i in range(2,20,1):
    clusterer = KMeans(n_clusters = i)
    clusterer.fit(reduced_data)

    # TODO: Predict the cluster for each data point
    preds = clusterer.predict(reduced_data)

    # TODO: Find the cluster centers
    centers = clusterer.cluster_centers_

    # TODO: Predict the cluster for each transformed sample data point
    sample_preds = clusterer.predict(pca_samples)

    # TODO: Calculate the mean silhouette coefficient for the number of clusters chosen
    score = silhouette_score(reduced_data, preds)
    print "i = %d, socre = %f" %(i,score)


# ### Question 7
# *Report the silhouette score for several cluster numbers you tried. Of these, which number of clusters has the best silhouette score?* 

# **Answer:**
# * The best score is 0.426281 when there are only 2 clusters. 

# ### Cluster Visualization
# Once you've chosen the optimal number of clusters for your clustering algorithm using the scoring metric above, you can now visualize the results by executing the code block below. Note that, for experimentation purposes, you are welcome to adjust the number of clusters for your clustering algorithm to see various visualizations. The final visualization provided should, however, correspond with the optimal number of clusters. 

# In[65]:

clusterer = KMeans(n_clusters = 2)
clusterer.fit(reduced_data)

# TODO: Predict the cluster for each data point
preds = clusterer.predict(reduced_data)

# TODO: Find the cluster centers
centers = clusterer.cluster_centers_

# TODO: Predict the cluster for each transformed sample data point
sample_preds = clusterer.predict(pca_samples)

# TODO: Calculate the mean silhouette coefficient for the number of clusters chosen
score = silhouette_score(reduced_data, preds)


# In[66]:

# Display the results of the clustering from implementation
vs.cluster_results(reduced_data, preds, centers, pca_samples)


# ### Implementation: Data Recovery
# Each cluster present in the visualization above has a central point. These centers (or means) are not specifically data points from the data, but rather the *averages* of all the data points predicted in the respective clusters. For the problem of creating customer segments, a cluster's center point corresponds to *the average customer of that segment*. Since the data is currently reduced in dimension and scaled by a logarithm, we can recover the representative customer spending from these data points by applying the inverse transformations.
# 
# In the code block below, you will need to implement the following:
#  - Apply the inverse transform to `centers` using `pca.inverse_transform` and assign the new centers to `log_centers`.
#  - Apply the inverse function of `np.log` to `log_centers` using `np.exp` and assign the true centers to `true_centers`.
# 

# In[67]:

# TODO: Inverse transform the centers
log_centers = pca.inverse_transform(centers)

# TODO: Exponentiate the centers
true_centers = np.exp(log_centers)

# Display the true centers
segments = ['Segment {}'.format(i) for i in range(0,len(centers))]
true_centers = pd.DataFrame(np.round(true_centers), columns = data.keys())
true_centers.index = segments
display(true_centers)
display(data.describe())


# ### Question 8
# Consider the total purchase cost of each product category for the representative data points above, and reference the statistical description of the dataset at the beginning of this project. *What set of establishments could each of the customer segments represent?*  
# **Hint:** A customer who is assigned to `'Cluster X'` should best identify with the establishments represented by the feature set of `'Segment X'`.

# **Answer:**
# * Segment 0:  
# Fresh - more than median,  
# Milk - less than median,  
# Grocery - around 25 percentile,  
# Frozen - more than 25 percentile,  
# Detergents_Pater - less than 25 percentile,  
# and Delicatesen - around 25 percentile. 
# * Segment 1:  
# Fresh - less than median,  
# Milk - more than 75 percentile,  
# Grocery - less than 75 percentile,  
# Frozen - around or less than 25 percentile,  
# Detergents_Pater - more than 75 percentile,  
# and Delicatesen - more than median. 

# ### Question 9
# *For each sample point, which customer segment from* ***Question 8*** *best represents it? Are the predictions for each sample point consistent with this?*
# 
# Run the code block below to find which cluster each sample point is predicted to be.

# In[68]:

# Display the predictions
for i, pred in enumerate(sample_preds):
    print "Sample point", i, "predicted to be in Cluster", pred


# **Answer:**
# * The prediction shows that all points are cluster 0. 
# * The result is not correct becasue there is one cluster here. Not usre what all points are in the same cluster. 

# ## Conclusion

# In this final section, you will investigate ways that you can make use of the clustered data. First, you will consider how the different groups of customers, the ***customer segments***, may be affected differently by a specific delivery scheme. Next, you will consider how giving a label to each customer (which *segment* that customer belongs to) can provide for additional features about the customer data. Finally, you will compare the ***customer segments*** to a hidden variable present in the data, to see whether the clustering identified certain relationships.

# ### Question 10
# Companies will often run [A/B tests](https://en.wikipedia.org/wiki/A/B_testing) when making small changes to their products or services to determine whether making that change will affect its customers positively or negatively. The wholesale distributor is considering changing its delivery service from currently 5 days a week to 3 days a week. However, the distributor will only make this change in delivery service for customers that react positively. *How can the wholesale distributor use the customer segments to determine which customers, if any, would react positively to the change in delivery service?*  
# **Hint:** Can we assume the change affects all customers equally? How can we determine which group of customers it affects the most?

# **Answer:**
# * The distributor can use segments to predict customer's purchasing categories. To use these segments information, distributor can follow the segments to prepare products. If costumers actually prucahse the similar amount of product followed by the certain categories, we can know that the reaction is positive.
# * If the reaction is positive, the distributor can predict changes between the 5 days a week and 3 days a week by using the segments information.

# ### Question 11
# Additional structure is derived from originally unlabeled data when using clustering techniques. Since each customer has a ***customer segment*** it best identifies with (depending on the clustering algorithm applied), we can consider *'customer segment'* as an **engineered feature** for the data. Assume the wholesale distributor recently acquired ten new customers and each provided estimates for anticipated annual spending of each product category. Knowing these estimates, the wholesale distributor wants to classify each new customer to a ***customer segment*** to determine the most appropriate delivery service.  
# *How can the wholesale distributor label the new customers using only their estimated product spending and the* ***customer segment*** *data?*  
# **Hint:** A supervised learner could be used to train on the original customers. What would be the target variable?

# **Answer:**
# * Apply the estimation annual spending to the unsupervised learning algorithm we used. For example, we use the K-means to get a customer segment model. We can use this to predict new customers. 

# ### Visualizing Underlying Distributions
# 
# At the beginning of this project, it was discussed that the `'Channel'` and `'Region'` features would be excluded from the dataset so that the customer product categories were emphasized in the analysis. By reintroducing the `'Channel'` feature to the dataset, an interesting structure emerges when considering the same PCA dimensionality reduction applied earlier to the original dataset.
# 
# Run the code block below to see how each data point is labeled either `'HoReCa'` (Hotel/Restaurant/Cafe) or `'Retail'` the reduced space. In addition, you will find the sample points are circled in the plot, which will identify their labeling.

# In[69]:

# Display the clustering results based on 'Channel' data
vs.channel_results(reduced_data, outliers, pca_samples)


# ### Question 12
# *How well does the clustering algorithm and number of clusters you've chosen compare to this underlying distribution of Hotel/Restaurant/Cafe customers to Retailer customers? Are there customer segments that would be classified as purely 'Retailers' or 'Hotels/Restaurants/Cafes' by this distribution? Would you consider these classifications as consistent with your previous definition of the customer segments?*

# **Answer:**
# * According the diagram, it looks like there are two groups. The algorithm we used has the highest silhouette coefficient when the number of clusters is two. It highly matches the distribution.
# * It is possible to have pure customer segments to represent 'Retailers' and 'Hotels/Restaurants/Cafes', but there will be overfitting in this case.
# *  It is pretty consistent of my segment prediction because the prediction shows there are two costumer segments.

# > **Note**: Once you have completed all of the code implementations and successfully answered each question above, you may finalize your work by exporting the iPython Notebook as an HTML document. You can do this by using the menu above and navigating to  
# **File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.
