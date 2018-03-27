function output_featureVector = fisher_proj(input_featureVector, input_labels, numClasses)
% input_featureVector - N x D input features of data
% input_labels - N x 1 labels of data
D = size(input_featureVector,2);
N = size(input_featureVector,1);

total_mean = mean(input_featureVector,1);

%computing class mean
class_mean = zeros(numClasses, D);
for i=1:numClasses
    idx = find(input_labels == categorical(i));
    data_i = input_featureVector(idx,:);
    class_mean(i,:) = sum(data_i,1) / size(data_i,1);
end

%Computing Sw
Sw = zeros(D);
Sk = zeros(D);
for i=1:numClasses
    idx = find(input_labels == categorical(i));
    data_i =input_featureVector(idx,:);
    for j=1:size(data_i,1)
        Sk = Sk + ((data_i(j,:) - class_mean(i,:))' *(data_i(j,:) - class_mean(i,:)));
    end
    Sw = Sw + Sk;
    Sk = zeros(D);
end

%Computing Sb
Sb = zeros(D);
for i=1:numClasses
    idx= find(input_labels == categorical(i));
    Nk = size(idx,1);
    Sb = Sb + Nk*((class_mean(i,:)-total_mean)'*(class_mean(i,:)-total_mean));
end

%Computing the projection
[eigenVector eigenValue] = eig(Sb,Sw,'qz');

%reduced dimension D'
num_of_dim = numClasses-1;
[eig_val eig_idx] = sort(diag(eigenValue),'descend');
W =eigenVector(:,eig_idx(1:num_of_dim));

output_featureVector = W;
end
