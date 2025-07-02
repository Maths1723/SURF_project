% 1. Load an image
img = imread('pubbl.png'); % Replace with your image

% Ensure image is grayscale for SURF detection
if size(img, 3) == 3
    gray_img = rgb2gray(img);
else
    gray_img = img;
end

% 2. Detect SURF features
% You can specify parameters like 'MetricThreshold' (default 1000)
% to control the number of detected features, or 'NumOctaves', 'NumScaleLevels'.
points = detectSURFFeatures(gray_img);

% Display the strongest 100 points with green crosses and circles
figure;
imshow(gray_img);
hold on;

% --- Manual Plotting of Features (with Green Crosses and Circles) ---
numPointsToDisplay = min(100, points.Count);
strongestPoints = points.selectStrongest(numPointsToDisplay);

if strongestPoints.Count > 0
    % Get Location, Scale, and Orientation for selected points
    locations = strongestPoints.Location; % [x, y] coordinates
    scales = strongestPoints.Scale;       % Characteristic scale
    
    % Plot green crosses at the feature locations
    plot(locations(:, 1), locations(:, 2), 'g+', 'MarkerSize', 10, 'LineWidth', 2);
    
    % Plot green circles for scale
    radii = scales * 6; % A factor of 6 is common for visual representation, given papers
    viscircles(locations, radii, 'Color', 'g', 'LineWidth', 0.5); % 'g' for green color

    fprintf('Manually plotted %d features with green crosses and circles.\n', strongestPoints.Count);
else
    fprintf('No strongest features to plot.\n');
end

title('Detected SURF Features (Green Crosses & Circles)');
hold off;

% 3. Extract SURF descriptors
% 'points' is the M-by-1 SURFPoints object returned by detectSURFFeatures.
% 'features' is an M-by-64 matrix of SURF descriptors.
% 'valid_points' is the subset of input points for which features were extracted.
[features, valid_points] = extractFeatures(gray_img, points);


% Example: Display one of the extracted features (not directly visualizable,
% but showing its structure)
disp('Size of features matrix:');
disp(size(features)); % Should be N x 64
disp('First 5 values of the first descriptor:');
disp(features(1, 1:5));

% % Match features:
% img2 = imread('pubbl.png');
% gray_img2 = rgb2gray(img2);
% points2 = detectSURFFeatures(gray_img2);
% [features2, valid_points2] = extractFeatures(gray_img2, points2);
% indexPairs = matchFeatures(features, features2);
% matchedPoints1 = valid_points(indexPairs(:,1));
% matchedPoints2 = valid_points2(indexPairs(:,2));
% 
% figure;
% showMatchedFeatures(gray_img, gray_img2, matchedPoints1, matchedPoints2, 'montage');
% title('Matched SURF Points');