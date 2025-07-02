% 1. Specify your image file
imagePath = 'elefanti.gif';

% Check if the file exists
if ~exist(imagePath, 'file')
    error('Image file not found: %s\nPlease ensure the image is in the current directory or provide its full path.', imagePath);
end

% 2. Load the image
img = imread(imagePath);

% Convert to grayscale if it's a color image (SURF usually works on grayscale)
if size(img, 3) == 3
    gray_img = rgb2gray(img);
else
    gray_img = img; % Already grayscale
end

% 3. Detect SURF features
% Lower threshold = more features.
points = detectSURFFeatures(gray_img, 'MetricThreshold', 500);

fprintf('Detected %d SURF features in %s\n', points.Count, imagePath);

% 4. Create a figure and use subplot to display images side-by-side
fig = figure('Name', ['Original vs. SURF Features: ' imagePath], 'NumberTitle', 'off', 'Units', 'normalized', 'Position', [0.1 0.1 0.8 0.7]); % Get figure handle

% Left subplot: Original Image
ax1 = subplot(1, 2, 1, 'Parent', fig);
imshow(img, 'Parent', ax1);
title(ax1, 'Original Image');
axis(ax1, 'on');

% Right subplot: Image with SURF Features
ax2 = subplot(1, 2, 2, 'Parent', fig);
imshow(gray_img, 'Parent', ax2);
hold(ax2, 'on');

fprintf('\n--- Debugging axes for feature plot ---\n');
disp(['Class of ax2: ', class(ax2)]);
disp(['Is ax2 valid: ', num2str(isvalid(ax2))]);
fprintf('Current axes (gca) before plot: %s\n', class(gca));
fprintf('Attempting manual plot of features...\n');


% Plot only the strongest features for clarity (e.g., top 100)
numPointsToDisplay = min(100, points.Count);
strongestPoints = points.selectStrongest(numPointsToDisplay);

% --- MANUAL PLOTTING OF FEATURE LOCATIONS ---
if strongestPoints.Count > 0
    x_coords = strongestPoints.Location(:, 1);
    y_coords = strongestPoints.Location(:, 2);

    % Plot green circles at the feature locations on ax2
    plot(ax2, x_coords, y_coords, 'go', 'MarkerSize', 5, 'LineWidth', 1.5);
    fprintf('Successfully plotted %d features manually.\n', strongestPoints.Count);
else
    fprintf('No strongest features to plot.\n');
end

title(ax2, sprintf('Image with %d Strongest SURF Features', numPointsToDisplay)); % Set title for ax2
hold(ax2, 'off');
axis(ax2, 'on');

% Add a super title for the entire figure
sgtitle(sprintf('Image Analysis: %s', imagePath));