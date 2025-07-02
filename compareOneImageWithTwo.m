function [allImagePoints, allImageFeatures, imageFileNames] = compareOneImageWithTwo(imageNameArray, refImageIdx, compImageIdx1, compImageIdx2)
    % compareOneImageWithTwo Detects and extracts SURF features for an array of images,
    % then compares one reference image with two specified comparison images.
    %
    %   [allImagePoints, allImageFeatures, imageFileNames] = compareOneImageWithTwo(imageNameArray, refImageIdx, compImageIdx1, compImageIdx2)
    %
    %   Inputs:
    %     imageNameArray - A cell array of strings, where each string is the
    %                      file path/name of an image.
    %     refImageIdx    - Index (1-based) of the reference image in imageNameArray.
    %     compImageIdx1  - Index (1-based) of the first comparison image in imageNameArray.
    %     compImageIdx2  - Index (1-based) of the second comparison image in imageNameArray.
    %
    %   Outputs:
    %     allImagePoints   - A cell array where each cell contains the
    %                        SURFPoints object for the corresponding image.
    %     allImageFeatures - A cell array where each cell contains the
    %                        M-by-64 SURF descriptors matrix for the
    %                        corresponding image.
    %     imageFileNames   - Returns the input imageNameArray for reference.

    if ~iscell(imageNameArray) || ~all(cellfun(@ischar, imageNameArray))
        error('Input `imageNameArray` must be a cell array of image file names (strings).');
    end

    numImages = length(imageNameArray);

    % Validate indices
    if ~isnumeric(refImageIdx) || ~isscalar(refImageIdx) || ...
       ~isnumeric(compImageIdx1) || ~isscalar(compImageIdx1) || ...
       ~isnumeric(compImageIdx2) || ~isscalar(compImageIdx2) || ...
       refImageIdx < 1 || refImageIdx > numImages || ...
       compImageIdx1 < 1 || compImageIdx1 > numImages || ...
       compImageIdx2 < 1 || compImageIdx2 > numImages
        error('Image indices (refImageIdx, compImageIdx1, compImageIdx2) must be valid 1-based indices within the image array.');
    end

    % Initialize cell arrays to store results for each image
    allImagePoints = cell(1, numImages);
    allImageFeatures = cell(1, numImages);

    fprintf('Starting SURF feature processing for %d images...\n', numImages);

    % --- Phase 1: Process all images to get features ---
    for i = 1:numImages
        currentImageName = imageNameArray{i};
        fprintf('\nProcessing image %d/%d: %s\n', i, numImages, currentImageName);

        try
            % 1. Load the image
            img = imread(currentImageName);

            % Ensure image is grayscale for SURF detection
            if size(img, 3) == 3
                gray_img = rgb2gray(img);
                fprintf('  Converted to grayscale.\n');
            else
                gray_img = img;
                fprintf('  Image is already grayscale.\n');
            end

            % 2. Detect SURF features
            points = detectSURFFeatures(gray_img);
            fprintf('  Detected %d SURF points.\n', points.Count);

            % 3. Extract SURF descriptors
            [features, valid_points] = extractFeatures(gray_img, points);
            fprintf('  Extracted %d SURF descriptors.\n', size(features, 1));

            % Store the results for the current image
            allImagePoints{i} = valid_points;
            allImageFeatures{i} = features;

            % Optional: Display the strongest points for the current image
            if points.Count > 0
                figure('Name', ['SURF Features for: ' currentImageName], 'NumberTitle', 'off');
                imshow(gray_img);
                hold on;
                numPointsToDisplay = min(20, points.Count);
                plot(points.selectStrongest(numPointsToDisplay));
                title(['Detected SURF Features for ' currentImageName ' (' num2str(numPointsToDisplay) ' strongest)']);
                hold off;
                drawnow; % Update figure immediately
            else
                fprintf('  No SURF features detected to display for this image.\n');
            end

        catch ME
            warning('Error processing image %s: %s\n', currentImageName, ME.message);
            % Store empty values for this image if an error occurs
            allImagePoints{i} = [];
            allImageFeatures{i} = [];
        end
    end

    fprintf('\nSURF processing complete for all images.\n');

    % --- Phase 2: Perform specified comparisons ---
    fprintf('\n--- Performing image comparisons ---\n');

    % Retrieve data for the comparison images
    refPts = allImagePoints{refImageIdx};
    refFeats = allImageFeatures{refImageIdx};
    refImgPath = imageNameArray{refImageIdx};

    comp1Pts = allImagePoints{compImageIdx1};
    comp1Feats = allImageFeatures{compImageIdx1};
    comp1ImgPath = imageNameArray{compImageIdx1};

    comp2Pts = allImagePoints{compImageIdx2};
    comp2Feats = allImageFeatures{compImageIdx2};
    comp2ImgPath = imageNameArray{compImageIdx2};

    % Check if features exist for the selected images
    if isempty(refFeats) || isempty(comp1Feats) || isempty(comp2Feats)
        warning('Cannot perform comparisons: Features not extracted for one or more of the specified images.');
        return; % Exit the function or skip comparisons
    end

    % Load actual images for display
    refImg = imread(refImgPath);
    if size(refImg, 3) == 3, refImg = rgb2gray(refImg); end

    comp1Img = imread(comp1ImgPath);
    if size(comp1Img, 3) == 3, comp1Img = rgb2gray(comp1Img); end

    comp2Img = imread(comp2ImgPath);
    if size(comp2Img, 3) == 3, comp2Img = rgb2gray(comp2Img); end


    % Comparison 1: Reference Image vs. Comparison Image 1
    fprintf('\nComparing %s with %s...\n', refImgPath, comp1ImgPath);
    indexPairs1 = matchFeatures(refFeats, comp1Feats);
    matchedPtsRef1 = refPts(indexPairs1(:,1));
    matchedPtsComp1 = comp1Pts(indexPairs1(:,2));

    figure('Name', ['Matches: ' refImgPath ' vs ' comp1ImgPath], 'NumberTitle', 'off');
    showMatchedFeatures(refImg, comp1Img, matchedPtsRef1, matchedPtsComp1, 'montage');
    title(sprintf('Matched SURF Points: %s vs %s (%d matches)', ...
                  strrep(refImgPath, '_', '\_'), strrep(comp1ImgPath, '_', '\_'), length(indexPairs1)));
    drawnow;

    % Comparison 2: Reference Image vs. Comparison Image 2
    fprintf('\nComparing %s with %s...\n', refImgPath, comp2ImgPath);
    indexPairs2 = matchFeatures(refFeats, comp2Feats);
    matchedPtsRef2 = refPts(indexPairs2(:,1));
    matchedPtsComp2 = comp2Pts(indexPairs2(:,2));

    figure('Name', ['Matches: ' refImgPath ' vs ' comp2ImgPath], 'NumberTitle', 'off');
    showMatchedFeatures(refImg, comp2Img, matchedPtsRef2, matchedPtsComp2, 'montage');
    title(sprintf('Matched SURF Points: %s vs %s (%d matches)', ...
                  strrep(refImgPath, '_', '\_'), strrep(comp2ImgPath, '_', '\_'), length(indexPairs2)));
    drawnow;

    fprintf('\nAll specified comparisons complete.\n');
end
