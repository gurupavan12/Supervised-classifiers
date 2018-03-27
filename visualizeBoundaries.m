% Visualizing Boundaries: this is using matlab's internal code
%         you will have to conform it your data structures to use
%     This is visualizing the linear function defining the class boundaries
%          boundaries for each class comparison (eq 4.9)
%    MDlLinear is a matrix of discriminant functions
%    featureVector is the features to show
%    labels are the feature vector labels as a categorical array
%    featureA and FeatureB are the index of the features to visualize
function visualizeBoundaries(w_0, w_1, w_2,featureVector,labels,featureA,featureB,categ)


clf
category_names = categories(labels);
numGroups = length(category_names);
colors = jet(numGroups*10);
colors = colors(round(linspace(1,numGroups*10,numGroups)),:);
x1_info =  [min(featureVector(:,featureA)),max(featureVector(:,featureA)),...
    min(featureVector(:,featureB)),max(featureVector(:,featureB))];

h1 = gscatter(featureVector(:,featureA),featureVector(:,featureB),labels,'','+o*v^');
for i = 1:numGroups
    h1(i).LineWidth = 2;
    h1(i).MarkerEdgeColor = min(colors(i,:)*1,1);
end
if categ == 1
    export_fig dataPoints_wine -png -transparent
elseif categ == 2
    export_fig dataPoints_wallpaper -png -transparent
elseif categ == 3
    export_fig dataPoints_taiji -png -transparent
else
    fprintf('None');
end

hold on
x =  [min(featureVector(:,featureA)),max(featureVector(:,featureA)) ];
h2 = [];
for i = 1:numGroups
    for j= 1:numGroups
        if i~=j
        wo{i} = w_0(i); wo{j} = w_0(j);
        w1{i} = w_1(i); w2{i} = w_2(i);
        w1{j} = w_1(j); w2{j} = w_2(j);
        h2 = cat(1,h2,plot(x, (wo{i} - wo{j} +(w1{i} - w1{j})*x)/(-(w2{i} - w2{j})) ,'LineWidth',1,...
            'DisplayName',sprintf('Class Sep b/w %s,%s',category_names{i},category_names{j})));
        end
    end
end



axis(x1_info);
hold off
grid on;
set(gca,'FontWeight','bold','LineWidth',2)
end