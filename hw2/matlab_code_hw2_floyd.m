%% Part 1 - Floyd - bass & guitar
clear all; close all; clc;

% load sound data
[S, Fs] = audioread('Floyd.m4a');
% p8 = audioplayer(S,Fs); playblocking(p8); % play the sound track

S = S(1: length(S)-1);
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
title('Sound Wave of Floyd clip');

subplot(2, 1, 2);
plot(ks, abs(fftshift(St))/max(abs(St)), 'r');
set(gca, 'XLim', [0 2000]);
xlabel('Frequency [Hz]');
ylabel('Amplitude');
title('The Fourier Transform');

%% Apply Gabor filter
close all;
% parameters for the filter (window)
a = 2000;
tau = 0:1:L;

for j = 1:length(tau)
    g = exp(-a*(t - tau(j)).^2); % Window function   
    Sg = g.*S;   
    Sgt = fft(Sg);
    
    % subplot(2, 1, 1);
    % plot(ks, abs(fftshift(Sgt))/max(abs(Sgt)), 'r');

    
    [M, I] = max(Sgt); % find the frequency with the maximum intensity
    
    % if the frequency with the maximum intensity is lower than 200Hz
    if 0 < k(I) && k(I) <= 200
        freq = k(I);
        % bass was played
        for i = 1:50 % remove overtune
            Sgt = Sgt.* (1-exp(-0.0004*(k-freq*i).^2)); % apply the filter around that frequency
        end
    end
    
    % subplot(2, 1, 2);

    % plot(ks, abs(fftshift(Sgt))/max(abs(Sgt)), 'r');
        
    Sgt = Sgt.*(1-exp(-0.00004*(k-90).^2)); % remove frequencies around 90Hz (bass)
    Sgt(length(Sgt)/2:length(Sgt)) = 0; % remove negative requencies

    [M, I] = max(Sgt); % find the frequency with the maximum intensity
    % k(I)
    Sgt = Sgt.*exp(-0.004*(k-k(I)).^2); % apply the filter around the frequency with maximum intensity

    % Sgt_notes(:,j) = abs(k(I)/2*pi); % record the maximum frequency
    Sgt_spec(:,j) = fftshift(abs(Sgt)); % We don't want to scale it
end
%% 
close all;
figure(2)
pcolor(tau,ks, Sgt_spec)
shading interp
set(gca,'ylim',[200 800],'Fontsize',16)
colormap(hot)
colorbar
xlabel('Time [sec]'), ylabel('Frequency [Hz]', 'Color', 'black')
title('The Spectrogram of Floyd Clip');

hold on;
y1 = yline(369,'--','F#4 369 Hz', 'LineWidth', 2, 'Color','white');
y1.LabelVerticalAlignment = 'bottom';
y1.LabelHorizontalAlignment = 'center';

y2 = yline(246,'--','B3 246 Hz', 'LineWidth', 2, 'Color', 'white');
y2.LabelVerticalAlignment = 'bottom';
y2.LabelHorizontalAlignment = 'center'; 

y3 = yline(587,'--','D5 587 Hz', 'LineWidth', 2, 'Color', 'white');
y3.LabelVerticalAlignment = 'bottom';
y3.LabelHorizontalAlignment = 'center';

y4 = yline(493,'--','B5 493 Hz', 'LineWidth', 2, 'Color', 'white');
y4.LabelVerticalAlignment = 'bottom';
y4.LabelHorizontalAlignment = 'center';

y5 = yline(440,'--','A4 440 Hz', 'LineWidth', 2, 'Color', 'white');
y5.LabelVerticalAlignment = 'bottom';
y5.LabelHorizontalAlignment = 'center';

y6 = yline(659,'--','E5 659 Hz', 'LineWidth', 2, 'Color', 'white');
y6.LabelVerticalAlignment = 'bottom';
y6.LabelHorizontalAlignment = 'center';

y7 = yline(329,'--','E4 329 Hz', 'LineWidth', 2, 'Color', 'white');
y7.LabelVerticalAlignment = 'bottom';
y7.LabelHorizontalAlignment = 'center';

% yticks([-1 -0.8 -0.2 0 0.2 0.8 1])
% ticklabels({'-3\pi','-2\pi','-\pi','0','\pi','2\pi','3\pi'})

% figure(3)
% plot(tau, Sgt_notes, 'o')