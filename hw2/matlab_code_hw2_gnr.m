%% Part 1 - GNR - guitar
clear all; close all; clc;

% load sound data
[S, Fs] = audioread('GNR.m4a');
% p8 = audioplayer(S,Fs); playblocking(p8); % play the sound track

% setup parameters
n = length(S);
t = (1:n) / Fs;
L = n / Fs;
k = (1/L)*[0:n/2-1 -n/2:-1]; ks = fftshift(k);


% vitualize the data
figure(1)
S = S'; St = fft(S);  
subplot(2, 1, 1);
plot(t, S);
xlabel('Time [sec]');
ylabel('Amplitude');
title('Sound Wave of GNR clip');

subplot(2, 1, 2);
plot(ks, abs(fftshift(St))/max(abs(St)), 'r');
set(gca, 'XLim', [0 2000]);
xlabel('Frequency [Hz]');
ylabel('Amplitude');
title('The Fourier Transform');

%% Apply Gabor filter

% parameters for the filter (window)
a = 2000;
tau = 0:0.1:L;

mm = 0;
mi = 0;
for j = 1:length(tau)   
    g = exp(-a*(t - tau(j)).^2); % Window function   
    Sg = g.*S;   
    Sgt = fft(Sg);
    
    [M, I] = max(Sgt); % find the frequency with the maximum intensity
    Sgt = Sgt.*exp(-0.000004*(k-300).^2); % apply the filter around that frequency
    
    Sgt_notes(:,j) = abs(k(I)/2*pi); % record the maximum frequency
    Sgt_spec(:,j) = fftshift(abs(Sgt)); % We don't want to scale it
end
%% Ploting
clc; close all;
figure(2)
pcolor(tau,ks, Sgt_spec)
shading interp
set(gca,'ylim',[0 600],'Fontsize',16)
colormap(hot)
colorbar
xlabel('Time [sec]'), ylabel('Frequency [Hz]', 'Color', 'black')
title('The Music Score of GNR Clip');

hold on;
y1 = yline(277,'--','C#4 277Hz', 'LineWidth', 2, 'Color','white');
y1.LabelVerticalAlignment = 'bottom';
y1.LabelHorizontalAlignment = 'center';

y2 = yline(311,'--','D#4 311Hz', 'LineWidth', 2, 'Color', 'white');
y2.LabelVerticalAlignment = 'bottom';
y2.LabelHorizontalAlignment = 'center';

y3 = yline(369,'--','F#4 369Hz', 'LineWidth', 2, 'Color', 'white');
y3.LabelVerticalAlignment = 'bottom';
y3.LabelHorizontalAlignment = 'center';

y4 = yline(415,'--','G#4 415Hz', 'LineWidth', 2, 'Color', 'white');
y4.LabelVerticalAlignment = 'bottom';
y4.LabelHorizontalAlignment = 'center';

y5 = yline(554,'--','C#5 415Hz', 'LineWidth', 2, 'Color', 'white');
y5.LabelVerticalAlignment = 'bottom';
y5.LabelHorizontalAlignment = 'center';
% figure(3)
% plot(tau, Sgt_notes, 'o')