This place is where I record my questions and what I've learnt from the project

## We have some outliers in the data, then what?
At first, my intention was use many unsupervised learning models to compare the robustness and then choose which models have better performance.
After detecting some outliers in our datasets, instead of handling them to specifically use distance-based models (k-means with Euclidean distance)
&rarr; I will use only PAM for clustering.

## Then why PAM but not CLARA?
After aggregating our datasets, we only have approximately 50 rows representing 50 stocks in the market, which is quite small and PAM can handle it well. CLARA applies for bigger datasets which take PAM forever to complete.

## pyclustertend.hopkins don't work with python:3.13, which is the current python I'm using for the project. Then I have to downgrade python version due to pyclustertend only works with python:3.10 to python:3.12. But then all the other dependencies crash after changing to python3:12
&rarr; Sadly, I have to install all over again