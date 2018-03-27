function [W_star,output_classifier_train, train_classificationMatrix, train_confusionMatrix, output_classifier_test, test_classificationMatrix, test_confusionMatrix, X_train, T_train, X_test, T_test,y_train, y_test] = linear_classification(feature_idx,numClasses,train_featureVector,train_labels,test_featureVector,test_labels)
 
%% Training
    train_featureVector = train_featureVector(:,feature_idx);
    N = size(train_featureVector,1);
    %Adding bias of 1 to the input matrix
    X_train = [ones(N,1) train_featureVector];
    
    %Target vector
    T_train = zeros(N, numClasses);
    for i=1:N
        j=train_labels(i);
        T_train(i,j) = 1;
    end
    
    %Calculating W* and Output vector
    W_star = (X_train'*X_train)\(X_train'*T_train);
    y_train = X_train*W_star;
    output_classifier_train(:,1) = zeros(N,1);
    for i=1:N
        [val pos] = max(abs(y_train(i,:)));
        output_classifier_train(i,1) = pos;
    end
    
    % Create confusion matrix
    train_confusionMatrix = confusionmat(train_labels, categorical(output_classifier_train));
    % Create classification matrix (rows should sum to 1)
    train_classificationMatrix = train_confusionMatrix./(meshgrid(countcats(train_labels))');

 %% Testing
    test_featureVector = test_featureVector(:,feature_idx);
    M = size(test_featureVector,1);
    
    %Adding bias of 1 to the input matrix
    X_test = [ones(M,1) test_featureVector];
    
    %Target vector
    T_test = zeros(M, numClasses);
    for i=1:M
        j=test_labels(i);
        T_test(i,j) = 1;
    end

    %Calculating Output vector
    y_test = X_test*W_star;
    output_classifier_test(:,1) = zeros(M,1);   
    for i=1:M
        [val pos] = max(abs(y_test(i,:)));
        output_classifier_test(i,1) = pos;
    end

    % Create confusion matrix
    test_confusionMatrix = confusionmat(test_labels, categorical(output_classifier_test));
    % Create classification matrix (rows should sum to 1)
    test_classificationMatrix = test_confusionMatrix./(meshgrid(countcats(test_labels))');



end
