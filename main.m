%code for project 2: linear classification
%pattern recognition, EE552
%Pavan Gurudath, Feb 2018
clc;
close all;
clear all;
addpath export_fig
% Choose which dataset to use (choices wine, wallpaper, taiji) :
dataset = {'wine', 'wallpaper', 'taiji'};
%% linear discriminate using least squares for classification
for i=1:3
    [train_featureVector, train_labels, test_featureVector, test_labels] = loadDataset(dataset{i});
    %% Linear Discriminant Classification using least squares

    %Using all the features
    %Training 
    numClasses = length(countcats(test_labels));
    
    %Sorting the data for training data
    [new_classNumber, ~] = grp2idx(train_labels); %Since classes are not numbers for wallpaper dataset
    new_train_featureVector = [new_classNumber train_featureVector];
    new_train_featureVector = sortrows(new_train_featureVector,1);
    train_featureVector = new_train_featureVector(:,2:end);
    train_labels = categorical(new_train_featureVector(:,1));

    %Sorting the data for testing data
    [new_classNumber, ~] = grp2idx(test_labels);
    new_test_featureVector = [new_classNumber test_featureVector];
    new_test_featureVector = sortrows(new_test_featureVector,1);
    test_featureVector = new_test_featureVector(:,2:end);
    test_labels = categorical(new_test_featureVector(:,1));

    feature_idx = 1:size(train_featureVector,2);
    
    [W_star,output_classifier_train, train_classificationMatrix,...
     train_confusionMatrix, output_classifier_test, test_classificationMatrix,...
     test_confusionMatrix, X_train, T_train, X_test, T_test,y_train,...
     y_test] = linear_classification(feature_idx,numClasses,...
            train_featureVector,train_labels, test_featureVector, test_labels);

    % mean group accuracy and std for training
    train_acc = mean(diag(train_classificationMatrix));
    train_std = std(diag(train_classificationMatrix));   
    fprintf(['The training accuracy for ',char(dataset{i}),' is ',num2str(train_acc*100),' percent \n']);
    % mean group accuracy and std for testing
    test_acc = mean(diag(test_classificationMatrix));
    test_std = std(diag(test_classificationMatrix));
    fprintf(['The testing accuracy for ',char(dataset{i}),' is ',num2str(test_acc*100),' percent \n']);
    
    switch dataset{i}
        case 'wine'    
            wine = struct('train_labels',train_labels,'test_labels',...
                test_labels,'train_featureVector', train_featureVector,...
                'test_featureVector', test_featureVector, 'X_train',...
                X_train,'X_test', X_test, 'T_train',T_train,'T_test',T_train,...
                'W_star',W_star,'y_train',y_train, 'y_test',y_test,...
                'output_classifier_train',output_classifier_train,...
                'output_classifier_test',output_classifier_test,...
                'train_classificationMatrix',train_classificationMatrix,...
                'train_confusionMatrix',train_confusionMatrix,...
                'test_classificationMatrix',test_classificationMatrix,...
                'test_confusionMatrix',test_confusionMatrix,...
                'test_acc',test_acc,'train_acc',train_acc,'test_std',...
                test_std, 'train_std', train_std,'feature_idx',feature_idx,...
                'numClasses', numClasses);        
            wine_weights = W_star;
            fprintf('\n-----------END--------------- \n');   
        case 'wallpaper'          
            wallpaper = struct('train_labels',train_labels,'test_labels',...
                test_labels,'train_featureVector', train_featureVector,...
                'test_featureVector', test_featureVector, 'X_train',...
                X_train,'X_test', X_test, 'T_train',T_train,'T_test',T_train,...
                'W_star',W_star,'y_train',y_train, 'y_test',y_test,...
                'output_classifier_train',output_classifier_train,...
                'output_classifier_test',output_classifier_test,...
                'train_classificationMatrix',train_classificationMatrix,...
                'train_confusionMatrix',train_confusionMatrix,...
                'test_classificationMatrix',test_classificationMatrix,...
                'test_confusionMatrix',test_confusionMatrix,...
                'test_acc',test_acc,'train_acc',train_acc,'test_std',...
                test_std, 'train_std', train_std,'feature_idx',feature_idx,...
                'numClasses', numClasses);
            wallpaper_weights = W_star;
            fprintf('\n-----------END--------------- \n');
        case 'taiji'
            taiji = struct('train_labels',train_labels,'test_labels',...
                test_labels,'train_featureVector', train_featureVector,...
                'test_featureVector', test_featureVector, 'X_train',...
                X_train,'X_test', X_test, 'T_train',T_train,'T_test',T_train,...
                'W_star',W_star,'y_train',y_train, 'y_test',y_test,...
                'output_classifier_train',output_classifier_train,...
                'output_classifier_test',output_classifier_test,...
                'train_classificationMatrix',train_classificationMatrix,...
                'train_confusionMatrix',train_confusionMatrix,...
                'test_classificationMatrix',test_classificationMatrix,...
                'test_confusionMatrix',test_confusionMatrix,...
                'test_acc',test_acc,'train_acc',train_acc,'test_std',...
                test_std, 'train_std', train_std,'feature_idx',feature_idx,...
                'numClasses', numClasses);        
            taiji_weights = W_star;
            fprintf('\n-----------END--------------- \n');
    end
end



%% PLOTTING THE GRAPHS
% Two features are used for the sake of being able to visualise the
% classification and thereby plot the graphs. 

%% Visualize wine classification by taking two features

fprintf('\n Wine dataset for 2 features \n')

featureA =1; featureB =7;
feature2_idx = [featureA featureB];

[W2_star, output2_classifier_train, train2_classificationMatrix, ...
    output2_classifier_test, test2_classificationMatrix, X2_train,...
    T2_train, X2_test, T2_test,y2_train, y2_test]...
    = linear_classification(feature2_idx,wine.numClasses,...
        wine.train_featureVector, wine.train_labels, ...
         wine.test_featureVector, wine.test_labels);

% mean group accuracy and std for training
train_acc = mean(diag(train2_classificationMatrix));
train_std = std(diag(train2_classificationMatrix));   

% mean group accuracy and std for testing
test_acc = mean(diag(test2_classificationMatrix));
test_std= std(diag(test2_classificationMatrix));

    
wine2 = struct('train_labels',wine.train_labels,'test_labels',...
            wine.test_labels,'train_featureVector', wine.train_featureVector,...
            'test_featureVector', wine.test_featureVector, 'X2_train',...
            X2_train,'X2_test', X2_test, 'T2_train',T2_train,'T2_test',...
            T2_test,'W2_star',W2_star,'y2_train',y2_train, 'y2_test',y2_test,...
            'output2_classifier_train',output2_classifier_train,...
            'output2_classifier_test',output2_classifier_test,...
            'train2_classificationMatrix', train2_classificationMatrix,...
            'test2_classificationMatrix',test2_classificationMatrix,...
            'test_acc',test_acc,'train_acc',train_acc,'test_std',...
            test_std,'train_std', train_std,'feature2_idx',feature2_idx);

wine2_weights = W2_star;

w_0 = wine2.W2_star(1,:);
w_1 = wine2.W2_star(2,:);
w_2 = wine2.W2_star(3,:);
entire_featureVector = [wine.test_featureVector];% wine.train_featureVector];
entire_labels = [wine.test_labels];% wine.train_labels];

figure(1);
visualizeBoundaries(w_0, w_1, w_2, entire_featureVector,entire_labels...
    ,featureA,featureB,1);
title('{\bf Linear Discriminant Classification for Wine}')
export_fig linear_discriminant_wine -png -transparent

fprintf('\n-----------END--------------- \n');  

%% Visualize wallpaper classification by taking two features

fprintf('\n Wallpaper dataset for 2 features \n')

featureA =1; featureB =7;

feature2_idx = [featureA featureB];

[W2_star, output2_classifier_train, train2_classificationMatrix, ...
    output2_classifier_test, test2_classificationMatrix, X2_train,...
    T2_train, X2_test, T2_test,y2_train, y2_test]...
    = linear_classification(feature2_idx,wallpaper.numClasses,...
        wallpaper.train_featureVector, wallpaper.train_labels, ...
         wallpaper.test_featureVector, wallpaper.test_labels);

% mean group accuracy and std for training
train_acc = mean(diag(train2_classificationMatrix));
train_std = std(diag(train2_classificationMatrix));   

% mean group accuracy and std for testing
test_acc = mean(diag(test2_classificationMatrix));
test_std= std(diag(test2_classificationMatrix));

    
wallpaper2 = struct('train_labels',wallpaper.train_labels,'test_labels',...
            wallpaper.test_labels,'train_featureVector', wallpaper.train_featureVector,...
            'test_featureVector', wallpaper.test_featureVector, 'X2_train',...
            X2_train,'X2_test', X2_test, 'T2_train',T2_train,'T2_test',...
            T2_test,'W2_star',W2_star,'y2_train',y2_train, 'y2_test',y2_test,...
            'output2_classifier_train',output2_classifier_train,...
            'output2_classifier_test',output2_classifier_test,...
            'train2_classificationMatrix', train2_classificationMatrix,...
            'test2_classificationMatrix',test2_classificationMatrix,...
            'test_acc',test_acc,'train_acc',train_acc,'test_std',...
            test_std,'train_std', train_std,'feature2_idx',feature2_idx);

wallpaper2_weights = W2_star;

w_0 = wallpaper2.W2_star(1,:);
w_1 = wallpaper2.W2_star(2,:);
w_2 = wallpaper2.W2_star(3,:);
entire_featureVector = [wallpaper.test_featureVector];% wallpaper.train_featureVector];
entire_labels = [wallpaper.test_labels];% wallpaper.train_labels];

figure(2);
visualizeBoundaries(w_0, w_1, w_2, entire_featureVector,entire_labels...
    ,featureA,featureB,2);
title('{\bf Linear Discriminant Classification for Wallpaper}')
export_fig linear_discriminant_wallpaper -png -transparent

fprintf('\n-----------END--------------- \n');   


%% Visualize taiji classification by taking two features

fprintf('\n Taiji dataset for 2 features \n')
featureA =1; featureB =7;
feature2_idx = [featureA featureB];

[W2_star, output2_classifier_train, train2_classificationMatrix, ...
    output2_classifier_test, test2_classificationMatrix, X2_train,...
    T2_train, X2_test, T2_test,y2_train, y2_test]...
    = linear_classification(feature2_idx,taiji.numClasses,...
        taiji.train_featureVector, taiji.train_labels, ...
         taiji.test_featureVector, taiji.test_labels);

% mean group accuracy and std for training
train_acc = mean(diag(train2_classificationMatrix));
train_std = std(diag(train2_classificationMatrix));   

% mean group accuracy and std for testing
test_acc = mean(diag(test2_classificationMatrix));
test_std= std(diag(test2_classificationMatrix));

    
taiji2 = struct('train_labels',taiji.train_labels,'test_labels',...
            taiji.test_labels,'train_featureVector', taiji.train_featureVector,...
            'test_featureVector', taiji.test_featureVector, 'X2_train',...
            X2_train,'X2_test', X2_test, 'T2_train',T2_train,'W2_star',...
            W2_star,'y2_train',y2_train, 'y2_test',y2_test,...
            'output2_classifier_train',output2_classifier_train,...
            'output2_classifier_test',output2_classifier_test,...
            'train2_classificationMatrix', train2_classificationMatrix,...
            'test2_classificationMatrix',test2_classificationMatrix,...
            'test_acc',test_acc,'train_acc',train_acc,'test_std',...
            test_std,'train_std', train_std,'feature2_idx',feature2_idx);

taiji2_weights = W2_star;

w_0 = taiji2.W2_star(1,:);
w_1 = taiji2.W2_star(2,:);
w_2 = taiji2.W2_star(3,:);
entire_featureVector = [taiji.test_featureVector];% taiji.train_featureVector];
entire_labels = [taiji.test_labels];% taiji.train_labels];

figure(3);
visualizeBoundaries(w_0, w_1, w_2, entire_featureVector,entire_labels...
    ,featureA,featureB,3);
title('{\bf Linear Discriminant Classification for Taiji}')
export_fig linear_discriminant_taiji -png -transparent

fprintf('\n-----------END--------------- \n');   

%% Fisher's linear discriminant 

clearvars -except wine wine2 taiji taiji2 wallpaper wallpaper2

%% Wine
wineFisher_W_featureVector= fisher_proj(wine.train_featureVector,wine.train_labels,wine.numClasses);
wineFisher_train_featureVector = wine.train_featureVector * wineFisher_W_featureVector;
wineFisher_test_featureVector = wine.test_featureVector * wineFisher_W_featureVector;

%knn Classifier
k=3;
pred_label = knnClassifier(wineFisher_train_featureVector,wine.train_labels,...
                wineFisher_test_featureVector, k); 
% Create confusion matrix
test_ConfMat = confusionmat(wine.test_labels,pred_label);
% Create classification matrix (rows should sum to 1)
test_ClassMat = test_ConfMat./(meshgrid(countcats(wine.test_labels))');
% mean group accuracy and std
test_acc = mean(diag(test_ClassMat));
test_std = std(diag(test_ClassMat));
fprintf(['Accuracy of kNN classifier for wine dataset is with mean ',num2str(test_acc),' and standard deviation ',num2str(test_std),'.\n']);

knn_wine = struct('predicted_label',pred_label,'Confusion_mat',...
            test_ConfMat,'Classification_mat',test_ClassMat,...
            'test_acc',test_acc,'test_std',test_std);
        
%% wallpaper
wallpaperFisher_W_featureVector= fisher_proj(wallpaper.train_featureVector,wallpaper.train_labels,wallpaper.numClasses);
wallpaperFisher_train_featureVector = wallpaper.train_featureVector * wallpaperFisher_W_featureVector;
wallpaperFisher_test_featureVector = wallpaper.test_featureVector * wallpaperFisher_W_featureVector;

%knn Classifier
k=3;
pred_label = knnClassifier(wallpaperFisher_train_featureVector,wallpaper.train_labels,...
                wallpaperFisher_test_featureVector, k); 
% Create confusion matrix
test_ConfMat = confusionmat(wallpaper.test_labels,pred_label);
% Create classification matrix (rows should sum to 1)
test_ClassMat = test_ConfMat./(meshgrid(countcats(wallpaper.test_labels))');
% mean group accuracy and std
test_acc = mean(diag(test_ClassMat));
test_std = std(diag(test_ClassMat));
fprintf(['Accuracy of kNN classifier for wallpaper dataset is with mean ',num2str(test_acc),' and standard deviation ',num2str(test_std),'.\n']);

knn_wallpaper = struct('predicted_label',pred_label,'Confusion_mat',...
            test_ConfMat,'Classification_mat',test_ClassMat,...
            'test_acc',test_acc,'test_std',test_std);

%% taiji
taijiFisher_W_featureVector= fisher_proj(taiji.train_featureVector,taiji.train_labels,taiji.numClasses);
taijiFisher_train_featureVector = taiji.train_featureVector * taijiFisher_W_featureVector;
taijiFisher_test_featureVector = taiji.test_featureVector * taijiFisher_W_featureVector;

%knn Classifier
k=3;
pred_label = knnClassifier(taijiFisher_train_featureVector,taiji.train_labels,...
                taijiFisher_test_featureVector, k); 
% Create confusion matrix
test_ConfMat = confusionmat(taiji.test_labels,pred_label);
% Create classification matrix (rows should sum to 1)
test_ClassMat = test_ConfMat./(meshgrid(countcats(taiji.test_labels))');
% mean group accuracy and std
test_acc = mean(diag(test_ClassMat));
test_std = std(diag(test_ClassMat));
fprintf(['Accuracy of kNN classifier for taiji dataset is with mean ',num2str(test_acc),' and standard deviation ',num2str(test_std),'.\n']);

knn_taiji = struct('predicted_label',pred_label,'Confusion_mat',...
            test_ConfMat,'Classification_mat',test_ClassMat,...
            'test_acc',test_acc,'test_std',test_std);
   
clearvars -except wine wine2 taiji taiji2 wallpaper wallpaper2 knn_wine ...
                knn_wallpaper knn_taiji

