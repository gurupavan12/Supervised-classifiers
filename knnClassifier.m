function accuracy = knnClassifier(train_featureVector,train_labels,test_featureVector, k)
    N = size(train_featureVector, 1);
    M = size(test_featureVector,1);
    count=0;
    
    for i=1:M
        distance = sqrt(sum((train_featureVector - repmat(test_featureVector(i,:), N, 1)).^2, 2));
        [val,ind] = sort(distance,'ascend');
        nearest_neigbrs = ind(1:k);
        x_closestLabels = train_labels(nearest_neigbrs);
        pred_label(i) = mode(x_closestLabels);
   end
   accuracy = pred_label';

end
