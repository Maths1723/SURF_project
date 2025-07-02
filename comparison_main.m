
% 1. Create the list of images.
image_list = {
    'base.jpeg',     ... % This will be our reference image (index 1)
    'close_up.jpeg', ... % This will be comparison image 1 (index 2)
    'rotated.jpeg'   ... % This will be comparison image 2 (index 3)
};

% 2. Define the indices for comparison.
%    We'll use 'base.jpeg' as the reference, and compare it with 'close_up.jpeg'
%    and 'rotated.jpeg'.
ref_idx = 1;      % Corresponds to 'base.jpeg'
comp1_idx = 2;    % Corresponds to 'close_up.jpeg'
comp2_idx = 3;    % Corresponds to 'rotated.jpeg'

% 3. Call the function to perform the processing and comparisons.
fprintf('Calling compareOneImageWithTwo with the specified images...\n');
[allPoints, allFeatures, processedFileNames] = compareOneImageWithTwo(image_list, ref_idx, comp1_idx, comp2_idx);

fprintf('\nFunction call completed. Check the generated figures for matches.\n');