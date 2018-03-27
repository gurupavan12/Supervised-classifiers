Note that every time the function is run, a new dataset is called. However, the data is being sorted initially, therefore the results are going to remain the same. 

- Run main.m
	This will call the following functions:
	- loadDataset.m  (Acquires the dataset)
	- linear_classification.m  (Extracts the feature Vector, labels and weights for testing and training)
	- fisher_proj.m   (Performs the Fisher projection)
	- knnClassifier.m  (Classification of data using kNN)

All the figures are saved in the working directory. 
The tables for Classification and Confusion matrix has been included in the submitted zip folder, however after execution they can be seen in the dataset's respective structures. 

The program has 9 structures where

- wine, wallpaper and taiji contains the attributes with regard to ALL the features/dimensions. 
- wine2, wallpaper2 and taiji2 contains the attributes with regard to ONLY TWO features/dimensions, namely 1 and 7.
- knn_wine, knn_wallpaper and knn_taiji contains the necessary attributes 




All the figures are generated and saved in the current working directory. 
