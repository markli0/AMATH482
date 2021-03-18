%% Clean workspace
clear all; close all; clc

%% setup
load subdata.mat % Imports the data as the 262144x49 (space by time) matrix called subdata

L = 10; % spatial domain
n = 64; % Fourier modes
x2 = linspace(-L,L,n+1); x = x2(1:n); y =x; z = x;
k = (2*pi/(2*L))*[0:(n/2 - 1) -n/2:-1]; ks = fftshift(k);

[X,Y,Z]=meshgrid(x,y,z);
[Kx,Ky,Kz]=meshgrid(ks,ks,ks);

%% problem 1
avg = zeros(64, 64, 64); % Initialize the average matrix
for j=1:49
    Un(:,:,:)=reshape(subdata(:,j),n,n,n); % data at time j
    
    M = max(abs(Un),[],'all');
    
    % Plot original data
    % isosurface(X,Y,Z,abs(Un) / M, 0.7);
    % axis([-20 20 -20 20 -20 20]); grid on;
    
    Unt = fftn(Un); % Transforms to frequency domain
    avg = avg + Unt; 
end
% average out at frequency domains
avg = fftshift(abs(avg)) / 49; 
M = max(abs(avg),[],'all');

% Plot the average of the spectrum
% isosurface(Kx,Ky,Kz,abs(avg) / M, 0.7);
% axis([-20 20 -20 20 -20 20]), grid on, drawnow

% calculate the center frequencies of every dimensions.
[f, v] = isosurface(Kx,Ky,Kz,abs(avg) / M, 0.7);
cent_freq = mean(v); 
x_cf = cent_freq(1);
y_cf = cent_freq(2);
z_cf = cent_freq(3);

%% problem 2
% Set up Gauss Filter 
a = 0.2;
x_gauss = exp(-a*(Kx-x_cf).^2); 
y_gauss = exp(-a*(Ky-y_cf).^2); 
z_gauss = exp(-a*(Kz-z_cf).^2); 
gauss_filter = x_gauss .* y_gauss .* z_gauss;

% Initialize position matrix
positions = zeros(49, 3);
for j=1:49
    Un(:,:,:)=reshape(subdata(:,j),n,n,n); % data at time j
    
    
    Unt = gauss_filter .* fftshift(fftn(Un)); % Applies filter on frequency domain
    Un = ifftn(Unt);% Transforms back to spacial domain
    
    % Plot denoised data
    % M = max(abs(Un),[],'all');
    % isosurface(X,Y,Z,abs(Un) / M, 0.7);
    % axis([-20 20 -20 20 -20 20]), grid on, drawnow
    
    % Calculate the center of the submarine at each time.
    [f, v] = isosurface(X,Y,Z,abs(Un) / M, 0.7);
    positions(j,:) = mean(v);
end

% Plot the path of the submarine
% plot3(positions(:,1), positions(:,2), positions(:,3), '-o','Color','b','MarkerSize',10,'MarkerFaceColor','#D9FFFF');

%% problem 3
r = table(positions(49, 1), positions(49, 2))

