%% clear all
clear all; close all; clc;

%% Setup
X = get_video('monte_carlo_low.mp4');
dt = 60;
t = linspace(0,size(X,2)/dt, size(X,2)-1);

%% Create DMD matrices

X1 = X(:,1:end-1);
X2 = X(:,2:end);

%% SVD of X1 and Computation of ~S

[U, Sigma, V] = svd(X1,'econ');

rank = 10;
U_l = U(:, 1:rank); % truncate to rank-r
Sigma_l = Sigma(1:rank, 1:rank);
V_l = V(:, 1:rank);
S = U_l'*X2*V_l*diag(1./diag(Sigma_l));
[eV, D] = eig(S); % compute eigenvalues + eigenvectors
mu = diag(D); % extract eigenvalues
omega = log(mu)/dt;
Phi = U_l*eV;

%% Create DMD Solution

b = Phi\X1(:,1); % pseudoinverse to get initial conditions

u_modes = zeros(length(b),length(t));
for iter = 1:length(t)
   u_modes(:,iter) = b.*exp(omega*t(iter)); 
end
u_dmd = Phi*u_modes;

%% Create Sparse and Low-Rank

Xsparse = X1 - abs(u_dmd);
clear X1; clear X2; clear U;
R = zeros(length(Xsparse), size(Xsparse, 2));

X_bg = R + abs(u_dmd);
X_fg = Xsparse - R;



%% Generate plots
frame = [80, 240];

for i=1:2
    org = imresize(rescale(reshape(X(:,frame(i)), 540, 960)),2);
    fg = imresize(rescale(reshape(X_fg(:,frame(i)), 540, 960)),2);
    bg = imresize(rescale(reshape(X_bg(:,frame(i)), 540, 960)),2);
    subplot(2,3,3*i-2);
    imshow(org);
    subplot(2,3,3*i-1);
    imshow(fg);
    subplot(2,3,3*i);
    imshow(bg);
end

%%
for i=1:size(X_bg,2)
   fg = rescale(reshape(X_bg(:,i), 540, 960));
   imshow(fg)
end



function mat = get_video(video_name)
obj = VideoReader(video_name);
vid = read(obj);
frames = obj.NumberOfFrames;
mat = [];
for x = 1 : frames
    frame = vid(:,:,:,x);
    frame = rgb2gray(frame);
    frame = double(frame(:));
    mat = [mat, frame];
end


end