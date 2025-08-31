% Path to your file
niftiFile = '/home/miguel/GI/2 - Reconstruction/scripts/Reconstructed-4D-original/4D_reconstructed.nii';
niftiData = niftiread(niftiFile);

% Extract the 4D MRI volume
stomachMask = niftiData; % Binary mask (1 = stomach, 0 = background)
[numRows, numCols, numSlices, numTimeFrames] = size(stomachMask);
disp('Size of stomachMask:');
disp(size(stomachMask)); % Should output [256, 256, Slices, TimeFrames]

%%%%%% Cross-Sectional Area Changes %%%%%%
crossSectionalAreas = zeros(numSlices, numTimeFrames);

for sliceIdx = 1:numSlices
    for t = 1:numTimeFrames
        % Sum all non-zero pixels in each slice
        currentSlice = squeeze(stomachMask(:, :, sliceIdx, t));
        crossSectionalAreas(sliceIdx, t) = sum(currentSlice(:));
    end
end
writematrix(crossSectionalAreas, 'crossSectionalAreas.csv');

%%%%%% Frequency Analysis %%%%%%
Fs = 1; % Sampling rate (frames per second)
physiologicalFreqRange = [0.05, 0.2]; % Hz range for gastric peristalsis

% Analyze mid-slice cross-sectional area changes
midSliceAreas = crossSectionalAreas(ceil(numSlices / 2), :);
writematrix(midSliceAreas, 'mid_slice_cross_sectional_area_uncentered.csv');

% 1-dimensional
detrendedData = midSliceAreas - mean(midSliceAreas); % Detrend data

% Export this data
writematrix(detrendedData, 'mid_slice_cross_sectional_area.csv');


% Apply FFT
Y = fft(detrendedData);
f = (0:(numTimeFrames - 1)) * Fs / numTimeFrames;
amplitudeSpectrum = abs(Y); 

% Filter to physiological range
frequencyMask = (f >= physiologicalFreqRange(1) & f <= physiologicalFreqRange(2));
filteredAmplitudes = amplitudeSpectrum .* frequencyMask;

% Identify dominant frequency
[~, dominantFreqIdx] = max(filteredAmplitudes);
dominantFreq = f(dominantFreqIdx);
disp('Dominant Frequency (Hz):');
disp(dominantFreq);

%%%%%% Wave Propagation Speed %%%%%%

% Cross-correlation between slices
waveSpeeds = zeros(1, numSlices - 1);
% for sliceIdx = 1:(numSlices - 1)
%     % Caution: This isn't mean centered without the detrending step
%     slice1 = crossSectionalAreas(sliceIdx, :);
%     slice2 = crossSectionalAreas(sliceIdx + 1, :);
    
%     % maxLag = 10;

%     % lags: amount slid from 1 to 2
%     % 
%     [xc, lags] = xcorr(slice1, slice2, 'none');
%     [~, maxIdx] = max(xc); % Find the lag with max correlation


%     % disp("Slice %d: best lag = %d frames (corr=%.3f)\n", sliceIdx, lags(maxIdx), xc(maxIdx));

%     timeLag = lags(maxIdx) / Fs; % Time delay in seconds
%     % Convert lag to speed (distance per time)
%     % waveSpeeds(sliceIdx) = distanceBetweenSlices / timeLag; % mm/s

%     if timeLag == 0
%         waveSpeeds(sliceIdx) = NaN;  % or skip this slice
%     else
%         waveSpeeds(sliceIdx) = distanceBetweenSlices / timeLag;
%     end
% end




distanceBetweenSlices = 5; % mm 

% Preallocate a struct array
debugInfo(numSlices-1) = struct( ...
    'sliceIdx',   [], ...
    'timeLag',    [], ...
    'xc',         [], ...
    'lags',       []  ...
);

for sliceIdx = 1:(numSlices - 1)
    slice1 = detrend(crossSectionalAreas(sliceIdx, :));
    slice2 = detrend(crossSectionalAreas(sliceIdx + 1, :));
    % raw cross-correlation
    [xc, lags] = xcorr(slice1, slice2, 'coeff');
    
    % Set 0-lag index to 0 to prevent it from being the max
    zeroLagIdx = find(lags == 0, 1);

    % If there is just one, set xc value to 0
    if ~isempty(zeroLagIdx)
        xc(zeroLagIdx) = 0;
    end
        
    [~, maxIdx] = max(xc);

    % compute timeLag
    timeLag = lags(maxIdx) / Fs;

    % store for debugging
    debugInfo(sliceIdx).sliceIdx = sliceIdx;
    debugInfo(sliceIdx).timeLag  = timeLag;
    debugInfo(sliceIdx).maxIdx  = maxIdx;
    debugInfo(sliceIdx).xc       = xc;    % these will become vectors in JSON
    debugInfo(sliceIdx).lags     = lags;
    
    % your existing speed logic...
    if timeLag == 0
        waveSpeeds(sliceIdx) = NaN;
    else
        waveSpeeds(sliceIdx) = distanceBetweenSlices / timeLag;
    end
end


% Convert to JSON and write to file
jsonStr = jsonencode(debugInfo, 'PrettyPrint', true);
fid = fopen('debug_results.json', 'w');
if fid == -1
    error('Cannot open debug_results.json for writing');
end
fwrite(fid, jsonStr, 'char');
fclose(fid);


disp('Wave propagation speeds (mm/s):');
disp(waveSpeeds);

%%%%%% Visualization %%%%%%
% 3D Visualization of Cross-Sectional Area with Time on X-Axis
figure;
[X, Y] = meshgrid(1:numSlices, 1:numTimeFrames); % Swap X and Y
surf(Y, X, crossSectionalAreas', 'EdgeColor', 'none'); % Transpose data to align axes
xlabel('Time Frame');
ylabel('Slice Index');
zlabel('Cross-Sectional Area (Pixels)');
title('3D Visualization of Peristalsis');
colormap jet;
colorbar;

%%%%%% Slice 35 Visualization %%%%%%
sliceToVisualize = 35; % Slice index to visualize

% Extract the cross-sectional area for the selected slice
slice35Areas = crossSectionalAreas(sliceToVisualize, :);

% Plot the changes in cross-sectional area over time
figure;
plot(1:numTimeFrames, slice35Areas, '-o', 'LineWidth', 2, 'MarkerSize', 5);
xlabel('Time Frame');
ylabel('Cross-Sectional Area (Pixels)');
title(sprintf('Cross-Sectional Area Changes for Slice %d', sliceToVisualize));
grid on;