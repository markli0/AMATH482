%% clear all
clear all; close all; clc;

%% Setup
load('cam1_2.mat')
load('cam2_2.mat')
load('cam3_2.mat')

filter1 = zeros(480,640);
filter1(200:400, 250:400) = 1;

filter2 = zeros(480,640);
filter2(50:370, 220:420) = 1;

filter3 = zeros(480,640);
filter3(190:350, 250:450) = 1;

%% analyze the video
videos = {vidFrames1_2, vidFrames2_2, vidFrames3_2};
filters = {filter1, filter2, filter3};
thres = [250, 250, 246];
data = {[], [], []};

min_frame = 10000;
for i = 1:3
    min_frame = min(size(videos{i},4), min_frame);
end

for i = 1:3
    video = videos{i};
    filter = filters{i};
    
    
    for j = 1:min_frame
    	img = video(:,:,:,j);

        img = rgb2gray(img);
        img = double(img);
        
        img = img.*filters{i};
        
        indeces = find(img > thres(i));
        [Y, X] = ind2sub(size(img), indeces);
        
        data{i} = [data{i}; mean(X), mean(Y)];
    end
end
%%
data = [data{1}'; data{2}'; data{3}'];

% switch the order of rows,
data = [data(1:4,:);data(6,:);data(5,:)];

%% make sure they are in phase
new_data = [];
for i = 1:3
    X = data(2*i-1,:);
    Y = data(2*i,:);

    [M, I] = max(Y(1:50));
    new_data = [new_data; X(I:I+150); Y(I:I+150)];
    
end
test = data;
data = new_data;

%% premodify the data
X = [];
for i = 1:6
   X = [X; data(i, :) - mean(data(i, :))];
end

%% apply the SVD
n = length(X(1,:));
A = X / sqrt(n-1);

[U,S,V] = svd(A, 'econ');

X_rank1 = sqrt(n-1)*U(:,1)*S(1,1)*V(:,1)';
X_rank2 = sqrt(n-1)*S(2,2)*U(:,2)*V(:,2)';

S = diag(S).^2;

%% plot
clc; close all;

set(groot,'defaultLineLineWidth',2.0)


for i=1:3
    subplot(3,2,2*i-1)
    plot(1:n,X(2*i-1,:),'r-','MarkerSize',10); hold on;
    plot(1:n,X(2*i,:),'c-','MarkerSize',10); 
    title(sprintf('camera %d - orginal displacement', i))
    xlim([0 180]);
    legend('XY', 'Z')
    xlabel('Time (Frames)')
    ylabel('Z-Displacement (Pixels)')

    subplot(3,2,2*i)
    plot(1:n,X_rank1(2*i,:),'b-','MarkerSize',10);  hold on;
    plot(1:n,X_rank2(2*i,:),'g-','MarkerSize',10); 
    title(sprintf('camera %d - approximation of displacement in z-axix', i))
    xlim([0 180]);
    xlabel('Time (Frames)')
    ylabel('Z-Displacement (Pixels)')

    legend('rank1', 'rank2')
    
end
set(findall(gcf,'-property','FontSize'),'FontSize',15)


figure(2)
p = pie(S);
pText = findobj(p,'Type','text');
percentValues = get(pText,'String'); 
% labels = {'rank1: ';'rank2: ';'rank3: ';'rank4: ';'rank5: ';'rank6: '};
labels = {'rank1: ';'rank2: ';' ';' ';'  ';'  '};
combinedtxt = strcat(labels,percentValues); 
pText(1).String = combinedtxt(1);
pText(2).String = combinedtxt(2);
pText(3).String = combinedtxt(3);
pText(4).String = combinedtxt(4);
pText(5).String = combinedtxt(5);
pText(6).String = combinedtxt(6);

figure(3)
for i=1:3
    subplot(3,1,i)
    plot(1:n,X(2*i,:),'k-','MarkerSize',10); hold on;
    plot(1:n,X_rank1(2*i,:),'c-','MarkerSize',10); 
    plot(1:n,X_rank1(2*i,:) + X_rank2(2*i,:) ,'r-','MarkerSize',10); 

    title(sprintf('camera %d - displacement in z-axis', i))
    xlim([0 180]);
    legend('original', 'rank1', 'rank1 + rank2')
    xlabel('Time (Frames)')
    ylabel('Z-Displacement (Pixels)')

end

set(findall(gcf,'-property','FontSize'),'FontSize',15)