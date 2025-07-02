% MAIN_surf_from_scratch.m
% Script to run SURF from scratch on pubbl.png, display top 20 keypoints, and compare with MATLAB SURF

clear; close all;

% Load the image
try
    img = imread('pubbl.png');
catch
    error('Failed to load pubbl.png. Ensure the file exists in the current directory or provide the full path.');
end

% Convert to grayscale if the image is RGB
if size(img, 3) == 3
    img = rgb2gray(img);
end
img = double(img) / 255; % Normalize to [0,1]
[height, width] = size(img);

% Display image properties
fprintf('Image size: %d x %d pixels\n', height, width);
fprintf('Image min/max intensity: %.4f / %.4f\n', min(img(:)), max(img(:)));

% Run custom SURF
try
    [keypoints, descriptors] = surf_from_scratch(img);
catch e
    error('Error running surf_from_scratch: %s', e.message);
end

% Sort keypoints by response and select top 20
max_features = 20;
if ~isempty(keypoints)
    responses = [keypoints.response];
    [~, sorted_idx] = sort(responses, 'descend');
    keypoints = keypoints(sorted_idx(1:min(max_features, length(keypoints))));
    descriptors = descriptors(sorted_idx(1:min(max_features, length(keypoints))), :);
end

% Debug: Display keypoint details, scale, and response distribution
fprintf('Custom SURF: Detected %d keypoints (displaying top %d)\n', length(keypoints), min(max_features, length(keypoints)));
if ~isempty(keypoints)
    fprintf('Top keypoint details (up to 5):\n');
    for i = 1:min(5, length(keypoints))
        kp = keypoints(i);
        fprintf('Keypoint %d: x=%.1f, y=%.1f, scale=%.2f, orientation=%.2f rad, response=%.6f\n', ...
                i, kp.x, kp.y, kp.scale, kp.orientation, kp.response);
    end
    scales = [keypoints.scale];
    responses = [keypoints.response];
    fprintf('Custom SURF: Scale range: %.2f to %.2f\n', min(scales), max(scales));
    fprintf('Custom SURF: Response range: %.6f to %.6f\n', min(responses), max(responses));
else
    warning('No keypoints detected. Try lowering the Hessian threshold or increasing filter sizes.');
end

% Visualize custom SURF keypoints
figure('Name', 'Top 20 Custom SURF Keypoints on pubbl.png', 'Position', [100, 100, 800, 600]);
imshow(img, []); % Auto-scale intensity
hold on;
for i = 1:length(keypoints)
    kp = keypoints(i);
    % Plot keypoint as a green circle
    plot(kp.x, kp.y, 'go', 'MarkerSize', max(10, kp.scale*30), 'LineWidth', 1.5);
    % Draw orientation as a green line
    len = max(15, kp.scale*30);
    x2 = kp.x + len * cos(kp.orientation);
    y2 = kp.y + len * sin(kp.orientation);
    plot([kp.x, x2], [kp.y, y2], 'g-', 'LineWidth', 1.5);
    % Label with scale
    text(kp.x+5, kp.y+5, sprintf('%.2f', kp.scale), 'Color', 'white', 'FontSize', 8);
end
title('Top 20 Custom SURF Keypoints on pubbl.png');
xlabel('X (pixels)'); ylabel('Y (pixels)');
axis on; axis tight;
hold off;
drawnow;

% Plot scale and response histograms for custom SURF
if ~isempty(keypoints)
    figure('Name', 'Custom SURF Distributions', 'Position', [100, 700, 800, 600]);
    subplot(2,1,1);
    histogram([keypoints.scale], 20);
    title('Custom SURF Keypoint Scale Distribution');
    xlabel('Scale'); ylabel('Count');
    subplot(2,1,2);
    histogram([keypoints.response], 20);
    title('Custom SURF Keypoint Response Distribution');
    xlabel('Response'); ylabel('Count');
end

% Run MATLAB SURF for comparison (requires Computer Vision Toolbox)
try
    matlab_keypoints = detectSURFFeatures(uint8(img*255));
    % Sort by response (Metric) and select top 20
    [~, sorted_idx] = sort(matlab_keypoints.Metric, 'descend');
    matlab_keypoints = matlab_keypoints(sorted_idx(1:min(max_features, length(matlab_keypoints))));
    
    fprintf('MATLAB SURF: Detected %d keypoints (displaying top %d)\n', length(matlab_keypoints), min(max_features, length(matlab_keypoints)));
    if ~isempty(matlab_keypoints)
        fprintf('MATLAB SURF: Top keypoint details (up to 5):\n');
        for i = 1:min(5, length(matlab_keypoints))
            kp = matlab_keypoints(i);
            fprintf('Keypoint %d: x=%.1f, y=%.1f, scale=%.2f, orientation=%.2f rad, metric=%.6f\n', ...
                    i, kp.Location(1), kp.Location(2), kp.Scale, kp.Orientation, kp.Metric);
        end
        fprintf('MATLAB SURF: Scale range: %.2f to %.2f\n', min(matlab_keypoints.Scale), max(matlab_keypoints.Scale));
        fprintf('MATLAB SURF: Metric range: %.6f to %.6f\n', min(matlab_keypoints.Metric), max(matlab_keypoints.Metric));
    end

    % Visualize MATLAB SURF keypoints
    figure('Name', 'Top 20 MATLAB SURF Keypoints on pubbl.png', 'Position', [900, 100, 800, 600]);
    imshow(img, []);
    hold on;
    for i = 1:length(matlab_keypoints)
        kp = matlab_keypoints(i);
        plot(kp.Location(1), kp.Location(2), 'go', 'MarkerSize', max(10, kp.Scale*30), 'LineWidth', 1.5);
        len = max(15, kp.Scale*30);
        x2 = kp.Location(1) + len * cos(kp.Orientation);
        y2 = kp.Location(2) + len * sin(kp.Orientation);
        plot([kp.Location(1), x2], [kp.Location(2), y2], 'g-', 'LineWidth', 1.5);
        text(kp.Location(1)+5, kp.Location(2)+5, sprintf('%.2f', kp.Scale), 'Color', 'white', 'FontSize', 8);
    end
    title('Top 20 MATLAB SURF Keypoints on pubbl.png');
    xlabel('X (pixels)'); ylabel('Y (pixels)');
    axis on; axis tight;
    hold off;
    drawnow;

    % Plot scale and response histograms for MATLAB SURF
    if ~isempty(matlab_keypoints)
        figure('Name', 'MATLAB SURF Distributions', 'Position', [900, 700, 800, 600]);
        subplot(2,1,1);
        histogram([matlab_keypoints.Scale], 20);
        title('MATLAB SURF Keypoint Scale Distribution');
        xlabel('Scale'); ylabel('Count');
        subplot(2,1,2);
        histogram([matlab_keypoints.Metric], 20);
        title('MATLAB SURF Keypoint Response Distribution');
        xlabel('Metric'); ylabel('Count');
    end
catch
    warning('MATLAB SURF failed. Ensure Computer Vision Toolbox is installed.');
end

% Save results
save('surf_results_top20.mat', 'keypoints', 'descriptors');

% Suggest parameter tuning if no large scales or weak responses
if ~isempty(keypoints)
    if max([keypoints.scale]) < 3
        disp('No large-scale keypoints detected. Suggestions:');
        disp('- Lower the Hessian threshold in surf_from_scratch.m (e.g., thresh = 0.00005 * (9/fs)).');
        disp('- Add larger filter sizes (e.g., append 87, 99 to filter_sizes).');
    end
    if max([keypoints.response]) < 0.001
        disp('Keypoint responses are weak. Suggestions:');
        disp('- Lower the Hessian threshold in surf_from_scratch.m (e.g., thresh = 0.00005 * (9/fs)).');
        disp('- Apply histogram equalization: img = histeq(img);');
    end
end