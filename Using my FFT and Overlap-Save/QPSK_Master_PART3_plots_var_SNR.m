%--Created by: Richard Cerrato--%
%--Date: 11/1/2023--%
%--Use ctrl-f and search 'PART[1-16]' to flip through different sections--%
%--Just click "Run", to test new random bit sequences, run the program again--%

sequence_len = 1000; % <--------- Change this to adjust length of b1 and b2 bit sequences




 for k = 0:5:10
    
    SNR = k; % Signal Noise Ratio
    
    
    
    
    %----------BIT SEQUENCE GENERATOR----------%
    %--PART1--%
    
    
    x1 = randn(1,sequence_len); % Generate a random seuqnce of numbers positve and negative with a zero-mean distribution of length sequnce_len 
    x2 = randn(1,sequence_len);
    
    b1 = zeros(size(sequence_len)); % Create an array of all zeros of the same size as the random sequence
    b2 = zeros(size(sequence_len));
    
    b1(x1 >= 0) = 1; % For every positive value of the random sequence, assign a +1 to the ne array
    b1(x1 < 0) = -1; % For every negative value of the random sequence, assign a -1 to the ne array
    
    b2(x2 >= 0) = 1;
    b2(x2 < 0) = -1;
    
    b1_upsampled = zeros(1, sequence_len * 4); % Create a new array with all zeros 4 times the size of the sequence
    b1_upsampled(1:4:end) = b1; % Replace the values of the bit sequence every 4th element
    
    b2_upsampled = zeros(1, sequence_len * 4);
    b2_upsampled(1:4:end) = b2;
    
    %---FOR PERFORMANCE EVAL--------------------------------------------------%
    sig_variance_1 = var(b1);
    sig_variance_2 = var(b2);
    disp(['Sig Var 1: ', num2str(sig_variance_1)]);
    disp(['Sig Var 2: ', num2str(sig_variance_2)]);
    
    noise_variance = sig_variance_1/(10^(SNR/10)); 
    disp(['Noise Var: ', num2str(noise_variance)]);
    noise_scalar = sqrt(noise_variance); % How much to scale the amplitude of the noise
    %---FOR PERFORMANCE EVAL--------------------------------------------------%
    
    % figure(1);
    % 
    % subplot(6,1,1)
    % plot(x1);
    % title('randn 1 (time)');
    % xlabel('t');
    % ylabel('A');
    % 
    % subplot(6,1,2)
    % stem(b1);
    % title('b1 (time)');
    % xlabel('t');
    % ylabel('A');
    % 
    % subplot(6,1,3)
    % stem(b1_upsampled);
    % title('b1 Upsampled x3 (time)');
    % xlabel('t');
    % ylabel('A');
    % 
    % subplot(6,1,4)
    % plot(x2);
    % title('randn 1 (time)');
    % xlabel('t');
    % ylabel('A');
    % 
    % subplot(6,1,5)
    % stem(b2);
    % title('b2 (time)');
    % xlabel('t');
    % ylabel('A');
    % 
    % subplot(6,1,6)
    % stem(b2_upsampled);
    % title('b2 Upsampled x3 (time)');
    % xlabel('t');
    % ylabel('A');
    
    
    %----------RAISED COSINE FUNCTION GENERATOR----------%
    %--PART2--%
    
    
    T = 4;
    B = 0.5;
    
    up_bound = (pi * (1 + B)) / T; % Establish the upper and lower bounds of the piecewise function
    down_bound = (pi * (1 - B)) / T;
    
    w = -pi:0.001*pi:pi;  % Set the sample rate and range
    
    val1 = sqrt(T); % Define the function at different parts of the piecewise
    
    val2 = sqrt( (T / 2) * (1 + cos((T / (2 * B)) * (abs(w) - down_bound))));
    
    Hrc = zeros(size(w)); % Create an array of zeros the size of the number of samples
    
    
    Hrc(abs(w) >= 0 & abs(w) <= down_bound) = val1; % Replace values with the outcome of the piecewise for each part depending on the bounds of w
    Hrc(abs(w) >= down_bound & abs(w) <= up_bound) = val2(abs(w) >= down_bound & abs(w) <= up_bound);
    
    
    hrc = ifft(fftshift(Hrc)); % Take the inverse fourier transform of Hrc, and center it, only plotting real values
    hrc = real(fftshift(hrc));
    
    
    start_index = max(1000 - 16, 1); % Find the center and the indexes of the values 16 to the let and right of the center
    end_index = min(1000 + 16, length(hrc));
    
    hrc_condensed= hrc(start_index:end_index); % create a new array that only includes 32 samples of hrc, 16 to right and left
    
    % figure(2);
    % 
    % subplot(3,1,1)
    % plot(Hrc)
    % title('Hrc (freq)');
    % xlabel('w');
    % ylabel('Mag');
    % 
    % 
    % subplot(3,1,2)
    % plot(hrc)
    % title('hrc (time)');
    % xlabel('t');
    % ylabel('A');
    % 
    % 
    % subplot(3,1,3)
    % stem(hrc_condensed)
    % title('hrc condensed x32 (time)');
    % xlabel('t');
    % ylabel('A');
    % 

    %----------CONVOLUTION (of b1 and b2 by hrc) VIA SLIDING-WINDOW----------%
    %--PART3--%

    b1_convolved = real(overlapSave(b1_upsampled,hrc_condensed));
    b2_convolved = real(overlapSave(b2_upsampled, hrc_condensed));

    % figure(3);
    % 
    % subplot(5,1,1)
    % stem(h)
    % title('hrc condensed x32 (time)');
    % xlabel('t');
    % ylabel('A');
    % 
    % subplot(5,1,2)
    % stem(b1_upsampled)
    % title('b1 Upsampled x3 (time)');
    % xlabel('t');
    % ylabel('A');
    % 
    % subplot(5,1,3)
    % %stem(b1_convolved)
    % plot(b1_convolved)
    % title('b1 * hrc (time)');
    % xlabel('t');
    % ylabel('A');
    % 
    % subplot(5,1,4)
    % stem(b2_upsampled)
    % title('b2 Upsampled x3 (time)');
    % xlabel('t');
    % ylabel('A');
    % 
    % subplot(5,1,5)
    % %stem(b2_convolved)
    % plot(b2_convolved)
    % title('b2 * hrc (time)');
    % xlabel('t');
    % ylabel('A');
    % 
    
    %----------UPSAMPLING AND LOWPASS FILTERING----------%
    %--PART4--%
    
    
    b1_conv_upsampled = zeros(1, length(b1_convolved) * 20);  % Create a new array with all zeros 20 times the size of the sequence
    b1_conv_upsampled(1:20:end) = b1_convolved; % Replace the values of the bit sequence every 20th element
    
    b2_conv_upsampled = zeros(1, length(b2_convolved) * 20); % Do the same for b2
    b2_conv_upsampled(1:20:end) = b2_convolved;
    
    b1_freq = real(fft(b1_conv_upsampled)); % Take the real values of the fourier transform of b1 and b2
    b2_freq = real(fft(b2_conv_upsampled));
    
    wl = 0.5*(375*2*pi)/(20*2000); % (3*pi)/(8*20) possibly multiply by 1/2
    
    lpf = fir1(50, wl, "low"); % create a lowpass filter with a cutoff frequency that matches the upsampled bandwidth ( 3pi/(8 * 20) )
    lpf = 20*lpf; % Scale the magnitude by 20, so the original magnitude can be recovered
    
    b1_filtered = filter(lpf, 1, b1_conv_upsampled); % Apply the lpf in time domain
    b2_filtered = filter(lpf, 1, b2_conv_upsampled);
    
    b1_freq_filtered = real(fftshift(fft(b1_filtered))); % Show the applied lpf to b1 and b2 in the frequency domain by taking the fourier transform
    b2_freq_filtered = real(fftshift(fft(b2_filtered)));
    
    
    % figure(4);
    % 
    % subplot(4,2,1)
    % plot(b1_conv_upsampled);
    % title('b1 Upsampled x20 (time)');
    % xlabel('t');
    % ylabel('A');
    % 
    % subplot(4,2,3)
    % semilogy(b1_freq);
    % title('b1 Upsampled (freq)');
    % xlabel('w');
    % ylabel('Mag');
    % 
    % subplot(4,2,5)
    % %semilogy(b1_freq_filtered);
    % plot(b1_freq_filtered);
    % title('b1 Upsampled lpf (freq)');
    % xlabel('w');
    % ylabel('Mag');
    % 
    % subplot(4,2,7)
    % plot(b1_filtered);
    % title('b1 Upsampled lpf (time)');
    % xlabel('t');
    % ylabel('A');
    % 
    % subplot(4,2,2)
    % plot(b2_conv_upsampled);
    % title('b2 Upsampled x20 (time)');
    % xlabel('t');
    % ylabel('A');
    % 
    % subplot(4,2,4)
    % semilogy(b2_freq);
    % title('b2 Upsampled (freq)');
    % xlabel('w');
    % ylabel('Mag');
    % 
    % subplot(4,2,6)
    % %semilogy(b2_freq_filtered);
    % plot(b2_freq_filtered);
    % title('b2 Upsampled lpf (freq)');
    % xlabel('w');
    % ylabel('Mag');
    % 
    % subplot(4,2,8)
    % plot(b2_filtered);
    % title('b2 Upsampled lpf (time)');
    % xlabel('t');
    % ylabel('A');
    
    
    %----------SINE AND COSINE MODULATION s[n]----------%
    %--PART5--%
    
    
    wc = (0.44 * pi) ; % Set the carrier frequency to 0.44pi
    
    %modulated_b1 = cos(wc * n + b1_filtered);
    %modulated_b2 = sin(wc * n + b2_filtered);
    
    %n = 0:4/(8640-1):4;
    
    modulated_b1 = cos(wc * (0:1:length(b1_filtered) - 1 )) .*b1_filtered; % Multiply both signals by their respective carrier signals, with the carrier signals scaled
    modulated_b2 = sin(wc * (0:1:length(b2_filtered) - 1 )) .*b2_filtered; % This is done in the time domain
    
    b1_freq_mod = real(fftshift(fft(modulated_b1))); % Take the fourier transofrm of these modulated signals, center them and plot real values
    b2_freq_mod = real(fftshift(fft(modulated_b2)));
    
    combined_sig = modulated_b1 + modulated_b2; % Add the two signals together to get one signal
    combined_sig_freq = real(fftshift(fft(combined_sig))); % Take the fourier transofrm
    
    
    step = 2 * pi / length(b1_freq_mod); % Set plotting parameters
    w = -pi:step:pi - step;
    
    % figure(5);
    % 
    % subplot(4, 2, 1);
    % plot(b1_filtered);
    % title('b1');
    % xlabel('Time');
    % ylabel('Amplitude');
    % 
    % subplot(4, 2, 3);
    % plot(modulated_b1);
    % title('Modulated b1');
    % xlabel('Time');
    % ylabel('Amplitude');
    % 
    % subplot(4, 2, 5);
    % plot(w, b1_freq_mod);
    % title('Modulated b1 (freq)');
    % xlabel('w');
    % ylabel('Mag');
    % 
    % subplot(4, 2, 7);
    % plot(combined_sig);
    % title('Combined Signal');
    % xlabel('Time');
    % ylabel('Amplitude');
    % 
    % subplot(4, 2, 2);
    % plot(b2_filtered);
    % title('b2');
    % xlabel('Time');
    % ylabel('Amplitude');
    % 
    % subplot(4, 2, 4);
    % plot(modulated_b2);
    % title('Modulated b2');
    % xlabel('Time');
    % ylabel('Amplitude');
    % 
    % subplot(4, 2, 6);
    % semilogy(w, b2_freq_mod);
    % title('Modulated b2 (freq)');
    % xlabel('w');
    % ylabel('Mag');
    % 
    % subplot(4, 2, 8);
    % semilogy(w, combined_sig_freq);
    % title('Combined Signal (freq)');
    % xlabel('w');
    % ylabel('Mag');
    % set(gca,'XTick',-pi:pi/4:pi);
    % set(gca,'XTickLabel',{'-\pi','-3\pi/4','-\pi/2','-\pi/4','0','\pi/4','\pi/2','3\pi/4','\pi'})
    
    
    %----------ADDITION OF NOISE----------%
    %--PART6--%
    
    
    noise = randn(1,length(combined_sig)); % Generate a random seuqnce of numbers positve and negative with a zero-mean distribution of length equal to the combined signals length
    
    noise = noise_scalar*noise; % Scale noise based on SNR
    
    noise_freq = real(fftshift(fft(noise))); % Take the fourier transform, center it and only look at real values
    
    transmitted_sig = combined_sig + noise; % Add the noise to the combined signal to simulate noise
    
    transmitted_sig_freq = real(fftshift(fft(transmitted_sig))); % Take the fourier transform, center it and only look at real values
    
    % figure(6);
    % 
    % subplot(3, 2, 1);
    % plot(combined_sig);
    % title('Combined Signal');
    % xlabel('Time');
    % ylabel('Amplitude');
    % 
    % subplot(3, 2, 2);
    % semilogy(w, combined_sig_freq);
    % title('Combined Signal (freq)');
    % xlabel('w');
    % ylabel('Mag');
    % set(gca,'XTick',-pi:pi/4:pi);
    % set(gca,'XTickLabel',{'-\pi','-3\pi/4','-\pi/2','-\pi/4','0','\pi/4','\pi/2','3\pi/4','\pi'})
    % 
    % subplot(3, 2, 3);
    % plot(noise);
    % title('Noise');
    % xlabel('Time');
    % ylabel('Amplitude');
    % 
    % subplot(3, 2, 4);
    % plot(w, noise_freq);
    % title('Noise (freq)');
    % xlabel('w');
    % ylabel('Mag');
    % set(gca,'XTick',-pi:pi/4:pi);
    % set(gca,'XTickLabel',{'-\pi','-3\pi/4','-\pi/2','-\pi/4','0','\pi/4','\pi/2','3\pi/4','\pi'})
    % 
    % subplot(3, 2, 5);
    % plot(transmitted_sig);
    % title('Signal with Noise');
    % xlabel('Time');
    % ylabel('Amplitude');
    % 
    % subplot(3, 2, 6);
    % plot(w, transmitted_sig_freq);
    % title('Signal with Noise (freq)');
    % xlabel('w');
    % ylabel('Mag');
    % set(gca,'XTick',-pi:pi/4:pi);
    % set(gca,'XTickLabel',{'-\pi','-3\pi/4','-\pi/2','-\pi/4','0','\pi/4','\pi/2','3\pi/4','\pi'})
    
    
    
    %--Created by: Richard Cerrato--%
    %--Date: 11/18/2023--%
    %--Use ctrl-f and search 'PART[1-6]' to flip through different sections--%
    %--Just click "Run" to excecute the program again. You must have all the--%
    %--included files in the same directory, they are the final output signals--%
    %--from Computing Assignment 1. If you wish to use your own inputs, you must--%
    %--name the files the same way I did--%
    
    %----------Bandpass Filter to Received Signal----------%
    %--PART7--%
    
    
    % Set the filter parameters
    cutoff_frequency = (3)/160; % Adjust the cutoff frequency
    center_frequency = 0.44*pi;
    filter_order = 300; % Adjust the filter order (CHANGE BACK TO 1000)
    
    % Define the time vector
    n = 0:filter_order;
    
    % Generate the bandpass filter
    h = (cutoff_frequency)*sinc(cutoff_frequency * (n - ((filter_order - 1)/2))) .* blackman(filter_order+1)';
    % Generate the cosine modulation
    cosine_modulation = 2*cos(center_frequency * n);
    
    % Apply the cosine modulation to shift the center frequency
    h_shifted = h .* cosine_modulation;
    %h_zerosa = h_shifted(1:length(transmitted_sig_freq));

    passed_sig = real(overlapSave(transmitted_sig, h_shifted));
    
    num_samples = length(transmitted_sig); % Reduces the outter edge of the bandpassed signal to be the length of the received signal
    passed_sig = passed_sig(1:num_samples); % This effectively removes the transient state
    
    passed_sig_freq = real(fftshift(fft(passed_sig))); % Takes the fourier transform to put the signal in the frequency domain
    h_freq = (fftshift(fft(h_shifted,length(transmitted_sig_freq))));
    
    % Test filtering by multiplying in frequ instead of convolving in time
    % --------------------------------------------------------------------
    
    test_filtered = h_freq.*transmitted_sig_freq;
    
    %---------------------------------------------------------------------
    
    freq_axis = linspace(-pi, pi, length(h_freq)); % This creates a variable for the frequency indexes that creates radian values that correspond to sample values
    
    % figure(7);
    % 
    % subplot(3,1,1);
    % plot(n, h);
    % title('Lowpass Filter using Sinc Function and Blackman Window');
    % xlabel('Time');
    % ylabel('Amplitude');
    % 
    % subplot(3,1,2);
    % plot(n, h_shifted);
    % title('Bandpass Filter using Sinc Function and Blackman Window');
    % xlabel('Time');
    % ylabel('Amplitude');
    % 
    % subplot(3,1,3);
    % plot(freq_axis, abs(h_freq));
    % title('Frequency Response of Bandpass Filter');
    % xlabel('w');
    % ylabel('Mag');
    % xlim([-pi, pi]);
    % set(gca,'XTick',-pi:pi/4:pi);
    % set(gca,'XTickLabel',{'-\pi','-3\pi/4','-\pi/2','-\pi/4','0','\pi/4','\pi/2','3\pi/4','\pi'});
    % 
    % figure(8);
    % 
    % subplot(3,2,1);
    % plot(n, h_shifted);
    % title('Bandpass Filter');
    % xlabel('Time');
    % ylabel('Amplitude');
    % 
    % subplot(3,2,3);
    % plot(transmitted_sig);
    % title('Received Signal');
    % xlabel('Time');
    % ylabel('Amplitude');
    % xlim([1, num_samples]);
    % 
    % % Display the frequency response
    % subplot(3,2,2);
    % plot(freq_axis, abs(h_freq));
    % title('Frequency Response of Bandpass Filter');
    % xlabel('w');
    % ylabel('Mag');
    % xlim([-pi, pi]);
    % set(gca,'XTick',-pi:pi/4:pi);
    % set(gca,'XTickLabel',{'-\pi','-3\pi/4','-\pi/2','-\pi/4','0','\pi/4','\pi/2','3\pi/4','\pi'});
    % 
    % subplot(3,2,4);
    % plot(freq_axis, abs(transmitted_sig_freq));
    % title('Received Signal (freq)');
    % xlabel('w');
    % ylabel('Mag');
    % xlim([-pi, pi]);
    % set(gca,'XTick',-pi:pi/4:pi);
    % set(gca,'XTickLabel',{'-\pi','-3\pi/4','-\pi/2','-\pi/4','0','\pi/4','\pi/2','3\pi/4','\pi'});
    % 
    % subplot(3,2,6);
    % plot(freq_axis, abs(passed_sig_freq));
    % title('Bandpassed Signal (freq)');
    % xlabel('w');
    % ylabel('Mag');
    % xlim([-pi, pi]);
    % set(gca,'XTick',-pi:pi/4:pi);
    % set(gca,'XTickLabel',{'-\pi','-3\pi/4','-\pi/2','-\pi/4','0','\pi/4','\pi/2','3\pi/4','\pi'});
    % 
    % subplot(3,2,5);
    % plot(passed_sig);
    % title('Bandpassed Signal');
    % xlabel('Time');
    % ylabel('Amplitude');
    % xlim([1, num_samples]);
    % 
    % figure(9);
    % 
    % plot(freq_axis, abs(test_filtered));
    % title('Multiplied Bandpass and Input in Freq');
    % xlabel('w');
    % ylabel('Mag');
    % xlim([-pi, pi]);
    % set(gca,'XTick',-pi:pi/4:pi);
    % set(gca,'XTickLabel',{'-\pi','-3\pi/4','-\pi/2','-\pi/4','0','\pi/4','\pi/2','3\pi/4','\pi'});
    
    
    %----------Downsampling Filtered Signal----------%
    %--PART8--%
    
    
    downsampled_val = 10; % Establishes the downsampling rate to be 10
    
    down_sampled_sig = zeros(1,(length(passed_sig)/downsampled_val)); % Creates a new array of length = the bandpass filtered signal divided by the downsample rate (10)
    
    for i = 1:length(passed_sig) % passes through every sample/value of the bandpassed signal
        modu = mod(i,downsampled_val);
        if modu == 0 % If the sample's index is divisble by 10 (10, 20, 30, 40...)
            down_sampled_sig(i/downsampled_val) = passed_sig(i); % Store the value of those ideces in the new array, one after the other
        end
    end
    
    num_samples = length(passed_sig);
    
    down_sampled_sig_freq = fftshift(fft(down_sampled_sig)); % Takes the fourier transform to put the signal in the frequency domain
    
    freq_axis = linspace(-pi, pi, length(passed_sig_freq));
    new_freq_axis = linspace(-pi, pi, length(down_sampled_sig_freq));
    
    % old center frequency = 0.44pi
    
    downsamp_center_freq = center_frequency*downsampled_val; % new center frequency = 0.44pi * 10 = 4.4pi
    
    while true % this is not centered in between 0 and 2pi 
        if downsamp_center_freq > (2*pi) % so we subtract 2pi from it until we get a value in between 0 and 2pi
            downsamp_center_freq = downsamp_center_freq - (2*pi); % this is done in this loop
        else 
            break; % Once the value is in between 0 and 2pi it stops, saving that value
        end
    end % 4.4pi -2pi -2pi = 0.4pi <- new carrier frequency
    
    
    % figure(10);
    % 
    % subplot(2,2,2);
    % plot(freq_axis, abs(passed_sig_freq));
    % title('Bandpassed Signal (freq)');
    % xlabel('w');
    % ylabel('Mag');
    % xlim([-pi, pi]);
    % set(gca,'XTick',-pi:pi/4:pi);
    % set(gca,'XTickLabel',{'-\pi','-3\pi/4','-\pi/2','-\pi/4','0','\pi/4','\pi/2','3\pi/4','\pi'});
    % 
    % subplot(2,2,1);
    % plot(passed_sig);
    % title('Bandpassed Signal');
    % xlabel('Time');
    % ylabel('Amplitude');
    % xlim([1, num_samples]);
    % 
    % subplot(2,2,4);
    % plot(new_freq_axis, abs(down_sampled_sig_freq));
    % title('Downsampled Signal (freq)');
    % xlabel('w');
    % ylabel('Mag');
    % xlim([-pi, pi]);
    % set(gca,'XTick',-pi:pi/4:pi);
    % set(gca,'XTickLabel',{'-\pi','-3\pi/4','-\pi/2','-\pi/4','0','\pi/4','\pi/2','3\pi/4','\pi'});
    % 
    % subplot(2,2,3);
    % plot(down_sampled_sig);
    % title('Downsampled Signal');
    % xlabel('Time');
    % ylabel('Amplitude');
    % xlim([1, length(down_sampled_sig)]);
    
    
    %----------Demodulating B1 and B2 Signals from combined Signal----------%
    %--PART9--%
    
    
    b1_demodulated = down_sampled_sig .* cos(downsamp_center_freq * (0:1:length(down_sampled_sig) - 1)); % Multiply the combined signal by a sine and cosine wave 
    b2_demodulated = down_sampled_sig .* sin(downsamp_center_freq * (0:1:length(down_sampled_sig) - 1)); % Both containing the new downasmpled center frequencies, this aextracts B1 and B2
    
    b1_demodulated_freq = fftshift(fft(b1_demodulated)); % Takes the fourier transform to put the signal in the frequency domain
    b2_demodulated_freq = fftshift(fft(b2_demodulated));
    
    demod_freq_axis = linspace(-pi, pi, length(b1_demodulated_freq)); 
    demod_num_samples = length(b1_demodulated);

    % figure(11);
    % 
    % subplot(3,2,1);
    % plot(down_sampled_sig);
    % title('Downsampled Signal');
    % xlabel('Time');
    % ylabel('Amplitude');
    % xlim([1, demod_num_samples]);
    % 
    % subplot(3,2,2);
    % plot(demod_freq_axis, abs(down_sampled_sig_freq));
    % title('Downsampled Signal (freq)');
    % xlabel('w');
    % ylabel('Mag');
    % xlim([-pi, pi]);
    % set(gca,'XTick',-pi:pi/4:pi);
    % set(gca,'XTickLabel',{'-\pi','-3\pi/4','-\pi/2','-\pi/4','0','\pi/4','\pi/2','3\pi/4','\pi'});
    % 
    % subplot(3,2,3);
    % plot(b1_demodulated);
    % title('B1 Demodulated By Cosine Signal');
    % xlabel('Time');
    % ylabel('Amplitude');
    % xlim([1, demod_num_samples]);
    % 
    % subplot(3,2,4);
    % plot(demod_freq_axis, abs(b1_demodulated_freq));
    % title('B1 Demodulated By Cosine Signal (freq)');
    % xlabel('w');
    % ylabel('Mag');
    % xlim([-pi, pi]);
    % set(gca,'XTick',-pi:pi/4:pi);
    % set(gca,'XTickLabel',{'-\pi','-3\pi/4','-\pi/2','-\pi/4','0','\pi/4','\pi/2','3\pi/4','\pi'});
    % 
    % subplot(3,2,5);
    % plot(b2_demodulated);
    % title('B2 Demodulated By Sine Signal');
    % xlabel('Time');
    % ylabel('Amplitude');
    % xlim([1, demod_num_samples]);
    % 
    % subplot(3,2,6);
    % plot(demod_freq_axis, abs(b2_demodulated_freq));
    % title('B2 Demodulated By Sine Signal (freq)');
    % xlabel('w');
    % ylabel('Mag');
    % xlim([-pi, pi]);
    % set(gca,'XTick',-pi:pi/4:pi);
    % set(gca,'XTickLabel',{'-\pi','-3\pi/4','-\pi/2','-\pi/4','0','\pi/4','\pi/2','3\pi/4','\pi'});
    
    
    %----------Lowpass Filtering the Demodulated Signals ----------%
    %--PART10--%
    
    
    downsamp_cutoff_freq = cutoff_frequency*10;
    % Defines the lowpass filtere as a sinc function with a cutoff frequency
    % equal to the original one scaled by the downsampled rate (10)
    lpf = (downsampled_val)*(downsamp_cutoff_freq)*sinc(downsamp_cutoff_freq * (n - ((filter_order - 1)/2))) .* blackman(filter_order+1)';
    

    b1_lpf = conv(b1_demodulated,lpf); %real(overlapSave(b1_demodulated, lpf));
    b2_lpf = conv(b2_demodulated,lpf);%real(overlapSave(b2_demodulated, lpf));

    
    b1_lpf_freq = fftshift(fft(b1_lpf)); % Takes the fourier transform to put the signal in the frequency domain
    b2_lpf_freq = fftshift(fft(b2_lpf)); % Takes the fourier transform to put the signal in the frequency domain
    
    demod_freq_axis = linspace(-pi, pi, length(b1_demodulated_freq)); % Sets a radian axis from -pi to pi that corresponds to the samples in the demodulated signals
    demod_num_samples = length(b1_demodulated); % stores the number of samples in the demodulated signals (b1 and b2 are the same)
    
    lpf_freq_axis = linspace(-pi, pi, length(b1_lpf_freq)); % Sets a radian axis from -pi to pi that corresponds to the samples in the filtered signals
    lpf_samples = length(b1_lpf);
    
    % figure(12);
    % 
    % subplot(3,2,1);
    % plot(down_sampled_sig);
    % title('Downsampled Signal');
    % xlabel('Time');
    % ylabel('Amplitude');
    % xlim([1, demod_num_samples]);
    % 
    % subplot(3,2,2);
    % plot(demod_freq_axis, abs(down_sampled_sig_freq));
    % title('Downsampled Signal (freq)');
    % xlabel('w');
    % ylabel('Mag');
    % xlim([-pi, pi]);
    % set(gca,'XTick',-pi:pi/4:pi);
    % set(gca,'XTickLabel',{'-\pi','-3\pi/4','-\pi/2','-\pi/4','0','\pi/4','\pi/2','3\pi/4','\pi'});
    % 
    % subplot(3,2,3);
    % plot(b1_demodulated);
    % title('B1 Demodulated By Cosine Signal');
    % xlabel('Time');
    % ylabel('Amplitude');
    % xlim([1, demod_num_samples]);
    % 
    % subplot(3,2,4);
    % plot(demod_freq_axis, abs(b1_demodulated_freq));
    % title('B1 Demodulated By Cosine Signal (freq)');
    % xlabel('w');
    % ylabel('Mag');
    % xlim([-pi, pi]);
    % set(gca,'XTick',-pi:pi/4:pi);
    % set(gca,'XTickLabel',{'-\pi','-3\pi/4','-\pi/2','-\pi/4','0','\pi/4','\pi/2','3\pi/4','\pi'});
    % 
    % subplot(3,2,5);
    % plot(b1_lpf);
    % title('B1 Demodulated lpf');
    % xlabel('Time');
    % ylabel('Amplitude');
    % xlim([1, lpf_samples]);
    % 
    % subplot(3,2,6);
    % plot(lpf_freq_axis, abs(b1_lpf_freq));
    % title('B1 Demodulated lpf (freq)');
    % xlabel('w');
    % ylabel('Mag');
    % xlim([-pi, pi]);
    % set(gca,'XTick',-pi:pi/4:pi);
    % set(gca,'XTickLabel',{'-\pi','-3\pi/4','-\pi/2','-\pi/4','0','\pi/4','\pi/2','3\pi/4','\pi'});
    % 
    % figure(13);
    % 
    % subplot(3,2,1);
    % plot(down_sampled_sig);
    % title('Downsampled Signal');
    % xlabel('Time');
    % ylabel('Amplitude');
    % xlim([1, demod_num_samples]);
    % 
    % subplot(3,2,2);
    % plot(demod_freq_axis, abs(down_sampled_sig_freq));
    % title('Downsampled Signal (freq)');
    % xlabel('w');
    % ylabel('Mag');
    % xlim([-pi, pi]);
    % set(gca,'XTick',-pi:pi/4:pi);
    % set(gca,'XTickLabel',{'-\pi','-3\pi/4','-\pi/2','-\pi/4','0','\pi/4','\pi/2','3\pi/4','\pi'});
    % 
    % subplot(3,2,3);
    % plot(b2_demodulated);
    % title('B2 Demodulated By Sine Signal');
    % xlabel('Time');
    % ylabel('Amplitude');
    % xlim([1, demod_num_samples]);
    % 
    % subplot(3,2,4);
    % plot(demod_freq_axis, abs(b2_demodulated_freq));
    % title('B2 Demodulated By Sine Signal (freq)');
    % xlabel('w');
    % ylabel('Mag');
    % xlim([-pi, pi]);
    % set(gca,'XTick',-pi:pi/4:pi);
    % set(gca,'XTickLabel',{'-\pi','-3\pi/4','-\pi/2','-\pi/4','0','\pi/4','\pi/2','3\pi/4','\pi'});
    % 
    % subplot(3,2,5);
    % plot(b2_lpf);
    % title('B2 Demodulated lpf');
    % xlabel('Time');
    % ylabel('Amplitude');
    % xlim([1, lpf_samples]);
    % 
    % subplot(3,2,6);
    % plot(lpf_freq_axis, abs(b2_lpf_freq));
    % title('B2 Demodulated lpf (freq)');
    % xlabel('w');
    % ylabel('Mag');
    % xlim([-pi, pi]);
    % set(gca,'XTick',-pi:pi/4:pi);
    % set(gca,'XTickLabel',{'-\pi','-3\pi/4','-\pi/2','-\pi/4','0','\pi/4','\pi/2','3\pi/4','\pi'});
    
    
    %----------Downsampling by 2 ----------%
    %--PART11--%
    
    
    b1_lpf = downsample(b1_lpf,2); % Downsample by factor of 2
    b2_lpf = downsample(b2_lpf,2);
    
    b1_lpf_freq = fftshift(fft(b1_lpf)); % Takes the fourier transform to put the signal in the frequency domain
    b2_lpf_freq = fftshift(fft(b2_lpf));
    
    downl_freq_axis = linspace(-pi, pi, length(b1_lpf_freq)); % Sets a radian axis from -pi to pi that corresponds to the samples in the filtered signals
    downl_samples = length(b1_lpf);
    
    figure(14);
    
    subplot(2,2,2);
    plot(downl_freq_axis, abs(b1_lpf_freq));
    title('B1 Downsampled by 2 (freq)');
    xlabel('w');
    ylabel('Mag');
    xlim([-pi, pi]);
    set(gca,'XTick',-pi:pi/4:pi);
    set(gca,'XTickLabel',{'-\pi','-3\pi/4','-\pi/2','-\pi/4','0','\pi/4','\pi/2','3\pi/4','\pi'});
    
    subplot(2,2,1);
    plot(b1_lpf);
    title('B1 Downsampled by 2');
    xlabel('Time');
    ylabel('Amplitude');
    xlim([1, downl_samples]);
    
    subplot(2,2,4);
    plot(downl_freq_axis, abs(b2_lpf_freq));
    title('B2 Downsampled by 2 (freq)');
    xlabel('w');
    ylabel('Mag');
    xlim([-pi, pi]);
    set(gca,'XTick',-pi:pi/4:pi);
    set(gca,'XTickLabel',{'-\pi','-3\pi/4','-\pi/2','-\pi/4','0','\pi/4','\pi/2','3\pi/4','\pi'});
    
    subplot(2,2,3);
    plot(b2_lpf);
    title('B2 Downsampled by 2');
    xlabel('Time');
    ylabel('Amplitude');
    xlim([1, downl_samples]);
    
    
    %--Created by: Richard Cerrato--%
    %--Date: 12/1/2023--%
    %--Computing Assignment 3--%
    
    
    %----------Match Filtering----------%
    %--PART12--%
    
    
    b1_matched = real((overlapSave(b1_lpf, hrc_condensed)));
    b2_matched = real((overlapSave(b2_lpf, hrc_condensed)));
    
    
    b1_matched_freq = fftshift(fft(b1_matched)); % Takes the fourier transform to put the signal in the frequency domain
    b2_matched_freq = fftshift(fft(b2_matched));
    
    matched_freq_axis = linspace(-pi, pi, length(b1_matched_freq)); % Sets a radian axis from -pi to pi that corresponds to the samples in the filtered signals
    matched_samples = length(b1_matched);
    
    figure(15);
    
    subplot(3,2,6)
    plot(hrc)
    title('hrc (time)');
    xlabel('t');
    ylabel('A');
    
    
    subplot(3,2,5)
    stem(hrc_condensed)
    title('hrc condensed x32 (time)');
    xlabel('t');
    ylabel('A');
    
    subplot(3,2,2);
    plot(matched_freq_axis, abs(b1_matched_freq));
    title('B1 Match Filtered (freq)');
    xlabel('w');
    ylabel('Mag');
    xlim([-pi, pi]);
    set(gca,'XTick',-pi:pi/4:pi);
    set(gca,'XTickLabel',{'-\pi','-3\pi/4','-\pi/2','-\pi/4','0','\pi/4','\pi/2','3\pi/4','\pi'});
    
    subplot(3,2,1);
    plot(b1_matched);
    title('B1 Match Filtered');
    xlabel('Time');
    ylabel('Amplitude');
    xlim([1, matched_samples]);
    
    subplot(3,2,4);
    plot(matched_freq_axis, abs(b2_matched_freq));
    title('B2 Match Filtered (freq)');
    xlabel('w');
    ylabel('Mag');
    xlim([-pi, pi]);
    set(gca,'XTick',-pi:pi/4:pi);
    set(gca,'XTickLabel',{'-\pi','-3\pi/4','-\pi/2','-\pi/4','0','\pi/4','\pi/2','3\pi/4','\pi'});
    
    subplot(3,2,3);
    plot(b2_matched);
    title('B2 Match Filtered');
    xlabel('Time');
    ylabel('Amplitude');
    xlim([1, downl_samples]);
    
    
    %----------Delay Estimation----------%
    %--PART13--%
    
    
    largest_delay = length(b1_matched)-length(b1_upsampled);
    
    c = zeros(1,largest_delay);
    
    for m = 1:(largest_delay)
    
        for n = 1:(length(b1_upsampled))
            c(m) = b1_upsampled(n)*b1_matched(n + m) + c(m);
        end
        
    end
    
    
    [maxVal, delay_idx] = max(c);
    
    disp(['Delay Value: ', num2str(delay_idx)]);
    
    b1_nodelay = circshift(b1_matched, -delay_idx);
    b2_nodelay = circshift(b2_matched, -delay_idx);
    
    figure(16);
    
    subplot(2,2,1);
    plot(b1_matched);
    title('B1 Matched');
    xlabel('Time');
    ylabel('Amplitude');
    
    subplot(2,2,2);
    plot(b2_matched);
    title('B2 Matched');
    xlabel('Time');
    ylabel('Amplitude');
    
    subplot(2,2,3);
    plot(b1_nodelay);
    title('B1 Delay Compensated');
    xlabel('Time');
    ylabel('Amplitude');
    
    subplot(2,2,4);
    plot(b2_nodelay);
    title('B2 Delay Compensated');
    xlabel('Time');
    ylabel('Amplitude');
    
    
    
    %----------Downsampling by 4----------%
    %--PART14--%
    
    
    downsampled_val = 4; % Establishes the downsampling rate to be 10
    
    b1_down_sampled = downsample(b1_nodelay, downsampled_val);
    b2_down_sampled = downsample(b2_nodelay, downsampled_val);
    
    b1_down_sampled_freq = fftshift(fft(b1_down_sampled)); % Takes the fourier transform to put the signal in the frequency domain
    b2_down_sampled_freq = fftshift(fft(b2_down_sampled)); % Takes the fourier transform to put the signal in the frequency domain
    
    downed_freq_axis = linspace(-pi, pi, length(b1_down_sampled_freq)); % Sets a radian axis from -pi to pi that corresponds to the samples in the filtered signals
    downed_samples = length(b1_down_sampled);
    
    figure(17);
    
    subplot(3,2,1);
    plot(b1_nodelay);
    title('B1 Delay Compensated');
    xlabel('Time');
    ylabel('Amplitude');
    
    subplot(3,2,2);
    plot(b2_nodelay);
    title('B2 Delay Compensated');
    xlabel('Time');
    ylabel('Amplitude');
    
    subplot(3,2,5);
    plot(downed_freq_axis, abs(b1_down_sampled_freq));
    title('B1 Downsampled by 4 (freq)');
    xlabel('w');
    ylabel('Mag');
    xlim([-pi, pi]);
    set(gca,'XTick',-pi:pi/4:pi);
    set(gca,'XTickLabel',{'-\pi','-3\pi/4','-\pi/2','-\pi/4','0','\pi/4','\pi/2','3\pi/4','\pi'});
    
    subplot(3,2,3);
    plot(b1_down_sampled);
    title('B1 Downsampled by 4');
    xlabel('Time');
    ylabel('Amplitude');
    xlim([1, downed_samples]);
    
    subplot(3,2,6);
    plot(downed_freq_axis, abs(b2_down_sampled_freq));
    title('B2 Downsampled by 4 (freq)');
    xlabel('w');
    ylabel('Mag');
    xlim([-pi, pi]);
    set(gca,'XTick',-pi:pi/4:pi);
    set(gca,'XTickLabel',{'-\pi','-3\pi/4','-\pi/2','-\pi/4','0','\pi/4','\pi/2','3\pi/4','\pi'});
    
    subplot(3,2,4);
    plot(b2_down_sampled);
    title('B2 Downsampled by 4');
    xlabel('Time');
    ylabel('Amplitude');
    xlim([1, downed_samples]);
    
    
    %----------Symbol Estimation----------%
    %--PART15--%
    
    
    b1_extracted = zeros(1, length(b1_down_sampled));
    b2_extracted = zeros(1, length(b2_down_sampled));
    
    for i = 1:length(b1_down_sampled)
        
        if b1_down_sampled(i) > 0
            b1_extracted(i) = 1;
        else
            b1_extracted(i) = -1;
        end
    
        if b2_down_sampled(i) > 0
            b2_extracted(i) = 1;
        else
            b2_extracted(i) = -1;
        end
    
    end
    
    extracted_samples = length(b1_extracted);
    
    figure(18);
    
    subplot(3,2,3);
    stem(b1_extracted);
    title('B1 Extracted');
    xlabel('Time');
    ylabel('Amplitude');
    xlim([1, sequence_len]);
    
    subplot(3,2,1);
    stem(b1);
    title('Original B1');
    xlabel('Time');
    ylabel('Amplitude');
    
    subplot(3,2,5);
    plot(b1_down_sampled);
    title('B1 Downsampled by 4');
    xlabel('Time');
    ylabel('Amplitude');
    xlim([1, sequence_len]);
    
    subplot(3,2,4);
    stem(b2_extracted);
    title('B2 Extracted');
    xlabel('Time');
    ylabel('Amplitude');
    xlim([1, sequence_len]);
    
    subplot(3,2,2);
    stem(b2);
    title('Original B2');
    xlabel('Time');
    ylabel('Amplitude');
    
    subplot(3,2,6);
    plot(b2_down_sampled);
    title('B2 Downsampled by 4');
    xlabel('Time');
    ylabel('Amplitude');
    xlim([1, sequence_len]);
    
    
    %----------Performance Evaluation----------%
    %--PART16--%A
    
    num_matching_samples = 0;
    
    for v = 1:sequence_len
        %check if both b1 matches b1_extracted AND b2 matches b2_extracted
        if (b1(v) == b1_extracted(v)) && (b2(v) == b2_extracted(v))
            num_matching_samples = num_matching_samples + 1;
        end
    end
    
    percent_error = ((sequence_len-num_matching_samples)/sequence_len)*100;
    
    disp(['Percent Error with a SNR of ', num2str(SNR), ' is: ', num2str(percent_error)]);

end

% SNR = 10*log10(var(sig)/var(noise))
% So var(noise) = var(sig) / ( 10^(SNR/10) )
% To get a desired variance from a randn() function, you scale the function by the
% sqrt(of the desired variance)
% scale_factor = sqrt(desired_variance)
% noise = scale_factor * randn(1,1000)



function y = overlapSave(x, h)
    M = length(h); % Determines length of filter h
    N = length(x); % Determines length of input signal x
    L = 2^nextpow2(N + M - 1) - M + 1; % Determines the length of the block for overlap-save
    r = rem(N, L); % Calculates remainder when N is divided by L
    x1 = [x zeros(1, L - r)]; % Zero-pads x to make its length a multiple of L
    num_segs = (length(x1)) / L; % Calculates the number of blocks
    h1 = [h zeros(1, L - 1)]; % Zero-pads filter h to match the block size

    % Loop through each block
    for k = 1:num_segs
        Ma(k, :) = x1(((k - 1) * L + 1):k * L); % Divides the signal into blocks
        if k == 1
            Ma1(k, :) = [zeros(1, M - 1) Ma(k, :)]; % Zero-pads the first block
        else
            Ma1(k, :) = [Ma(k - 1, (L - M + 2):L) Ma(k, :)]; % Overlaps and concatenates blocks
        end
        Ma2(k, :) = ifft(myFFT2_zpdng(Ma1(k, :)) .* myFFT2_zpdng(h1)); % Performs convolution in frequency domain
    end

    Ma3 = Ma2(:, M:(L + M - 1)); % Removes extra elements from convolution result
    y1 = Ma3'; % Transposes the result
    y = y1(:)'; % Reshapes the result into a row vector
end


function X = myFFT2_zpdng(x)
    
    N = length(x);
    
    if mod(log2(N),1) ~= 0
        num_zpads = 2^(ceil(log2(N))) - N;
        x = [x zeros(1,num_zpads)];
    end
    
    X = myFFT2(x);

end


function X = myFFT2(x)
    N = length(x);

    if (N==2)
        X(1) = x(1) + x(2);
        X(2) = x(1) - x(2);
    else
        X1 = myFFT2(x(1:2:N));
        X2 = myFFT2(x(2:2:N));
        X = [X1, X1] + exp(-1i*2*pi/N*(0:N-1)).*[X2, X2];
    end

end

function X = myiFFT2_zpdng(x)
    
    N = length(x);
    
    if mod(log2(N),1) ~= 0
        num_zpads = 2^(ceil(log2(N))) - N;
        x = [x zeros(1,num_zpads)];
    end
    
    X = myiFFT2(x)./length(x);
    
end


function X = myiFFT2(x)
    N = length(x);

    if (N==2)
        X(1) = x(1) + x(2);
        X(2) = x(1) - x(2);
    else
        X1 = myFFT2(x(1:2:N));
        X2 = myFFT2(x(2:2:N));
        X = [X1, X1] + exp(1i*2*pi/N*(0:N-1)).*[X2, X2];
    end

end
