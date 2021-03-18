clear all; close all; clc;

[data, train_labels] = mnist_parse('train-images.idx3-ubyte', 'train-labels.idx1-ubyte');

data = double(data);
data = data(:)';
train_images = reshape(data(:), [784, 60000]);
% train_images = train_images - repmat(mean(train_images, 1), 784, 1);


[data, test_labels] = mnist_parse('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte');

data = double(data);
data = data(:)';
test_images = reshape(data(:), [784, 10000]);
% test_images = test_images - repmat(mean(test_images, 1), 784, 1);

    %% part 1.1-2
    [U, S, V] = svd(train_images, 'econ');

    %% check U
    for k = 1:25
        subplot(5,5,k)
        u = reshape(U(:,k), 28, 28);
        u = rescale(u);
        imshow(u)
    end

%% check S
close all;
s = diag(S);

figure(1)
subplot(2,1,1)
plot(s, 'ko','Linewidth',1)
ylabel('\sigma')
title('\sigma after applying SVD on the training set')

subplot(2,1,2)
semilogy(s,'ko','Linewidth',1)
ylabel('\sigma (log scale)')
title('\sigma after applying SVD on the training set (log scale)')
%% V
basis = [2,3,5];
digits = [];
for i = 1:10
   digit_label = find(train_labels == i-1);
   digits = [digits; digit_label(1:10)'];
end

v = V(:, basis);
for i = 1:10
    plot3(v(digits(i,:), 1), v(digits(i,:),2), v(digits(i,:),3), 'o'); hold on;
    xlabel('second principle component')
    ylabel('third principle component')
    zlabel('fifth principle component')
end

%% part 2.1
close all;
d1 = get_images_by_label(0, train_images, train_labels);
d2 = get_images_by_label(1, train_images, train_labels);

dt1 = get_images_by_label(0, test_images, test_labels);
dt2 = get_images_by_label(1, test_images, test_labels);

[U, w, threshold] = lda2d(d1, d2, 700);
accur = lda2d_test(dt1, dt2, U, w, threshold);

%% part 2.2
close all; clc;
d1 = get_images_by_label(0, train_images, train_labels);
d2 = get_images_by_label(1, train_images, train_labels);
d3 = get_images_by_label(2, train_images, train_labels);
[U, w, t1, t2] = lda3d(d1, d2, d3, 700);

%% part 2.3-4
% calculation
accur = ones(10);
for i = 1:10
    for j = 1:10
        if i == j
            continue
        end
        d1 = get_images_by_label(i-1, train_images, train_labels);
        d2 = get_images_by_label(j-1, train_images, train_labels);

        dt1 = get_images_by_label(i-1, test_images, test_labels);
        dt2 = get_images_by_label(j-1, test_images, test_labels);
        
        [U, w, threshold] = lda2d(d1, d2, 700);
        accur(i, j) = lda2d_test(d1, d2, U, w, threshold);
    end
end
%% plot
load('DTC_accur.mat');
load('SVM_accur.mat');

accur = SVM_accur;
[X,Y] = meshgrid(0:9,0:9);
surf(X, Y, accur);
colorbar
xlabel('first digit')
ylabel('second digit')
zlabel('accuracy')

%% find the min and amx
[Min, I] = min(accur(:));
[min_d1, min_d2] = ind2sub(size(accur),I);
accur_d = accur - diag(ones(1,10));
[Max, I] = max(accur_d(:));
[max_d1, max_d2] = ind2sub(size(accur_d),I);

%% part 2.5 DTC and SVM
DTC_accur = ones(10);
SVM_accur = ones(10);
for i = 0:9
    for j = 0:9
        if i == j
            continue
        end
        dtr1 = get_images_by_label(i, train_images, train_labels);
        dtr2 = get_images_by_label(j, train_images, train_labels);

        dte1 = get_images_by_label(i, test_images, test_labels);
        dte2 = get_images_by_label(j, test_images, test_labels);
        
        d_train = [dtr1, dtr2];
        l_train = [ones(1, size(dtr1, 2)) * (i), ones(1, size(dtr2, 2)) * (j)];

        d_test = [dte1, dte2];
        l_test = [ones(1, size(dte1, 2)) * (i), ones(1, size(dte2, 2)) * (j)];
        
        % DTC
        DTC = fitctree(d_train', l_train);
        p = predict(tree, d_test')';
        DTC_accur(i+1, j+1) = sum((p - l_test) == 0) / size(d_test, 2);

        % SVM
        %SVM = fitcsvm(d_train', l_train);
        %p = predict(SVM, d_test')';
        %SVM_accur(i, j) = sum((p - l_test) == 0) / size(d_test, 2);
    end
end



function mat = get_images_by_label(digit,images,labels)
    indices = labels == digit;
    mat = images(:, indices);
end

function [U, w,threshold] = lda2d(d1,d2,feature)
    n1 = size(d1,2);
    n2 = size(d2,2);

    [U,S,V] = svd([d1 d2],'econ');
    
    d = S*V';
    U = U(:,1:feature);
    
    d1 = d(1:feature, 1:n1);
    d2 = d(1:feature, n1+1:n1+n2);
    m1 = mean(d1,2);
    m2 = mean(d2,2);

    d1s = d1 - m1;
    d2s = d2 - m2;
    Sw = d1s*d1s' + d2s * d2s';
    Sb = (m1-m2)*(m1-m2)'; % between class

    [V2, D] = eig(Sb,Sw); % linear disciminant analysis
    [lambda, ind] = max(abs(diag(D)));
    w = V2(:,ind);
    w = w/norm(w,2);
    
    v1 = w'*d1;
    v2 = w'*d2;

    if mean(v1) > mean(v2)
        w = -w;
        v1 = -v1;
        v2 = -v2;
    end
    %figure(4)
    %plot(v1,0,'ob')
    %hold on
    %plot(v2,1,'dr')
    %ylim([0 1])

    % Find the threshold value

    s1 = sort(v1);
    s2 = sort(v2);

    t1 = length(s1);
    t2 = 1;
    while s1(t1) > s2(t2)
        t1 = t1 - 1;
        t2 = t2 + 1;
    end
    
    threshold = (s1(t1) + s2(t2))/2;

    % Plot histogram of results
    %figure(6)
    %subplot(1,2,1)
    %histogram(s1,30); hold on, plot([threshold threshold], [0 1400],'r')
    %title('Train: digit 0')
    %subplot(1,2,2)
    %histogram(s2,30); hold on, plot([threshold threshold], [0 3000],'r')
    %title('Train: digit 1')
end


function [U, w, t1, t2] = lda3d(d1,d2,d3, feature)
    n1 = size(d1,2);
    n2 = size(d2,2);
    n3 = size(d3,2);


    [U,S,V] = svd([d1 d2 d3],'econ');
    
    d = S*V';
    U = U(:,1:feature);
    
    d1 = d(1:feature, 1:n1);
    d2 = d(1:feature, n1+1:n1+n2);
    d3 = d(1:feature, n1+n2+1:n1+n2+n3);

    m1 = mean(d1,2);
    m2 = mean(d2,2);
    m3 = mean(d3,2);
    m = mean([d1,d2,d3],2);
    
    Sw = 0;
    for k = 1:n1
        Sw = Sw + (d1(:,k) - m1)*(d1(:,k) - m1)';
    end
    for k = 1:n2
        Sw =  Sw + (d2(:,k) - m2)*(d2(:,k) - m2)';
    end
    for k = 1:n3
        Sw =  Sw + (d3(:,k) - m3)*(d3(:,k) - m3)';
    end

    Sb = (m1-m)*(m1-m)' + (m2-m)*(m2-m)' + (m3-m)*(m3-m)';

    [V2, D] = eig(Sb,Sw); % linear disciminant analysis
    [lambda, ind] = max(abs(diag(D)));
    w = V2(:,ind);
    w = w/norm(w,2);
    
    v1 = w'*d1;
    v2 = w'*d2;
    v3 = w'*d3;


    plot(v1,0,'ob','Linewidth',2)
    hold on
    plot(v2,1,'dr','Linewidth',2)
    plot(v3,2,'ok','Linewidth',2)

    ylim([0 2.2])

    % Find the threshold value

    s1 = sort(v1);
    s2 = sort(v2);
    s3 = sort(v3);


    t1 = length(s3);
    t2 = 1;
    while s2(t1) > s3(t2)
        t1 = t1 - 1;
        t2 = t2 + 1;
    end
    
    th1 = (s2(t1) + s3(t2))/2;
    
    t1 = length(s1);
    t2 = 1;
    while s3(t1) > s1(t2)
        t1 = t1 - 1;
        t2 = t2 + 1;
    end
    
    th2 = (s3(t1) + s1(t2))/2;
    
    figure(5)
    subplot(1,3,1)
    histogram(s1,30); hold on, plot([th2 th2], [0 1000],'r')
    title('Train: digit 0')
    subplot(1,3,2)
    histogram(s2,30); hold on, plot([th1 th1], [0 2500],'r')
    title('Train: digit 1')
    subplot(1,3,3)
    histogram(s3,30); hold on, plot([th2 th2], [0 1000],'r'); plot([th1 th1], [0 1000],'r')
    title('Train: digit 2')
    
end


function accur = lda2d_test(d1, d2, U, w, threshold)
    feature = size(w, 1);
    
    n1 = size(d1,2);
    n2 = size(d2,2);
    
    d = U'*[d1 d2];
    U = U(:,1:feature);
    
    d1 = d(1:feature, 1:n1);
    d2 = d(1:feature, n1+1:n1+n2);
    
    s1 = sort(w'*d1);
    s2 = sort(w'*d2);
    
    count = sum(s1 < threshold) + sum(s2 > threshold);
    accur = count / (n1+n2);
    

    
end


function [imgs_train, labels] = mnist_parse(path_to_digits, path_to_labels)

% The function is curtesy of stackoverflow user rayryeng from Sept. 20,
% 2016. Link: https://stackoverflow.com/questions/39580926/how-do-i-load-in-the-mnist-digits-and-label-data-in-matlab

% Open files
fid1 = fopen(path_to_digits, 'r');

% The labels file
fid2 = fopen(path_to_labels, 'r');

% Read in magic numbers for both files
A = fread(fid1, 1, 'uint32');
magicNumber1 = swapbytes(uint32(A)); % Should be 2051
fprintf('Magic Number - imgs_train: %d\n', magicNumber1);

A = fread(fid2, 1, 'uint32');
magicNumber2 = swapbytes(uint32(A)); % Should be 2049
fprintf('Magic Number - Labels: %d\n', magicNumber2);

% Read in total number of imgs_train
% Ensure that this number matches with the labels file
A = fread(fid1, 1, 'uint32');
totalimgs_train = swapbytes(uint32(A));
A = fread(fid2, 1, 'uint32');
if totalimgs_train ~= swapbytes(uint32(A))
    error('Total number of imgs_train read from imgs_train and labels files are not the same');
end
fprintf('Total number of imgs_train: %d\n', totalimgs_train);

% Read in number of rows
A = fread(fid1, 1, 'uint32');
numRows = swapbytes(uint32(A));

% Read in number of columns
A = fread(fid1, 1, 'uint32');
numCols = swapbytes(uint32(A));

fprintf('Dimensions of each digit: %d x %d\n', numRows, numCols);

% For each image, store into an individual slice
imgs_train = zeros(numRows, numCols, totalimgs_train, 'uint8');
for k = 1 : totalimgs_train
    % Read in numRows*numCols pixels at a time
    A = fread(fid1, numRows*numCols, 'uint8');

    % Reshape so that it becomes a matrix
    % We are actually reading this in column major format
    % so we need to transpose this at the end
    imgs_train(:,:,k) = reshape(uint8(A), numCols, numRows).';
end

% Read in the labels
labels = fread(fid2, totalimgs_train, 'uint8');

% Close the files
fclose(fid1);
fclose(fid2);

end
