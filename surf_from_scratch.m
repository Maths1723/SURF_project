% SURF from Scratch in MATLAB (Proof-of-Concept, Enhanced for Relevant Keypoints)
% Input: Grayscale image (uint8 or double)
% Output: Keypoints (struct with x, y, scale, orientation) and 64D descriptors

function [keypoints, descriptors] = surf_from_scratch(image)
    % Convert image to double and normalize
    if ~isfloat(image)
        image = double(image) / 255;
    end
    [height, width] = size(image);

    % Preprocess: Histogram equalization to enhance contrast
    image = histeq(image);

    % Step 1: Compute Integral Image
    integral_img = integralImage(image);

    % Step 2: Detect Keypoints using Hessian
    filter_sizes = [9, 15, 21, 27]; % Max scale ~3.67
    scales = filter_sizes / 9; % Scale normalization
    keypoints = struct('x', {}, 'y', {}, 'scale', {}, 'response', {});
    detH_maps = cell(1, length(filter_sizes)); % Store Hessian maps
    keypoints_per_scale = zeros(1, length(filter_sizes)); % Track keypoints per scale

    for i = 1:length(filter_sizes)
        fs = filter_sizes(i);
        s = scales(i);

        % Compute Hessian responses (Lxx, Lyy, Lxy)
        [Lxx, Lyy, Lxy] = hessianResponses(integral_img, fs, height, width);
        detH = Lxx .* Lyy - 0.81 * Lxy.^2; % Hessian determinant
        detH = detH / (s^4); % Normalize by scale^2
        detH_maps{i} = detH;

        % Non-maximum suppression within scale
        thresh = 0.001 * (9/fs)^4; % Stricter threshold
        [rows, cols] = find(detH > thresh);
        for j = 1:length(rows)
            r = rows(j); c = cols(j);
            if isLocalMax(detH, r, c, height, width, fs)
                keypoints(end+1) = struct('x', c, 'y', r, 'scale', s, 'response', detH(r,c));
                keypoints_per_scale(i) = keypoints_per_scale(i) + 1;
            end
        end
    end

    % Debug: Display keypoints per scale
    fprintf('Keypoints per scale:\n');
    for i = 1:length(filter_sizes)
        fprintf('Scale %.2f (filter size %d): %d keypoints\n', scales(i), filter_sizes(i), keypoints_per_scale(i));
    end

    % Cross-scale non-maximum suppression
    keypoints = crossScaleSuppression(keypoints, detH_maps, filter_sizes, height, width);

    % Spatial clustering removal
    keypoints = removeClusteredKeypoints(keypoints, filter_sizes);

    % Step 3: Assign Orientations
    for i = 1:length(keypoints)
        kp = keypoints(i);
        keypoints(i).orientation = computeOrientation(integral_img, kp.x, kp.y, kp.scale, height, width);
    end

    % Step 4: Compute Descriptors
    descriptors = zeros(length(keypoints), 64);
    for i = 1:length(keypoints)
        kp = keypoints(i);
        descriptors(i, :) = computeDescriptor(integral_img, kp.x, kp.y, kp.scale, kp.orientation, height, width);
    end
end

% Helper: Compute Integral Image
function int_img = integralImage(img)
    int_img = cumsum(cumsum(img, 2), 1);
    int_img = [zeros(1, size(img,2)+1); [zeros(size(img,1),1), int_img]];
end

% Helper: Compute Hessian Responses
function [Lxx, Lyy, Lxy] = hessianResponses(int_img, filter_size, height, width)
    Lxx = zeros(height, width);
    Lyy = zeros(height, width);
    Lxy = zeros(height, width);
    half_size = floor(filter_size / 2);

    for r = half_size+1:height-half_size
        for c = half_size+1:width-half_size
            Lxx(r, c) = boxFilter(int_img, r, c, half_size, 'xx');
            Lyy(r, c) = boxFilter(int_img, r, c, half_size, 'yy');
            Lxy(r, c) = boxFilter(int_img, r, c, half_size, 'xy');
        end
    end
end

% Helper: Box Filter for Hessian
function response = boxFilter(int_img, r, c, half_size, type)
    if strcmp(type, 'xx')
        w = floor(half_size / 3);
        x1 = c - half_size; x2 = c + half_size;
        y1 = r - w; y2 = r + w;
        pos = boxIntegral(int_img, x1, y1, x2, y2);
        neg1 = boxIntegral(int_img, x1, r-w-2*w, x2, r-w);
        neg2 = boxIntegral(int_img, x1, r+w, x2, r+w+2*w);
        response = pos - (neg1 + neg2);
    elseif strcmp(type, 'yy')
        w = floor(half_size / 3);
        x1 = c - w; x2 = c + w;
        y1 = r - half_size; y2 = r + half_size;
        pos = boxIntegral(int_img, x1, y1, x2, y2);
        neg1 = boxIntegral(int_img, c-w-2*w, y1, c-w, y2);
        neg2 = boxIntegral(int_img, c+w, y1, c+w+2*w, y2);
        response = pos - (neg1 + neg2);
    else % Lxy
        s = floor(half_size / 2);
        pos1 = boxIntegral(int_img, c-s, r-s, c, r);
        pos2 = boxIntegral(int_img, c, r, c+s, r+s);
        neg1 = boxIntegral(int_img, c, r-s, c+s, r);
        neg2 = boxIntegral(int_img, c-s, r, c, r+s);
        response = (pos1 + pos2) - (neg1 + neg2);
    end
end

% Helper: Box Integral (sum over rectangle)
function sum_val = boxIntegral(int_img, x1, y1, x2, y2)
    x1 = max(1, x1); y1 = max(1, y1);
    x2 = min(size(int_img,2)-1, x2);
    y2 = min(size(int_img,1)-1, y2);
    sum_val = int_img(y2+1, x2+1) - int_img(y2+1, x1) - int_img(y1, x2+1) + int_img(y1, x1);
end

% Helper: Non-Maximum Suppression (within scale)
function is_max = isLocalMax(detH, r, c, height, width, filter_size)
    is_max = true;
    half_size = max(2, floor(filter_size / 2)); % Larger neighborhood
    for dr = -half_size:half_size
        for dc = -half_size:half_size
            if dr == 0 && dc == 0
                continue;
            end
            rr = r + dr; cc = c + dc;
            if rr < half_size+1 || rr > height-half_size || cc < half_size+1 || cc > width-half_size
                continue;
            end
            if detH(rr, cc) >= detH(r, c)
                is_max = false;
                return;
            end
        end
    end
end

% Helper: Cross-Scale Non-Maximum Suppression
function keypoints = crossScaleSuppression(keypoints, detH_maps, filter_sizes, height, width)
    keep = true(1, length(keypoints));
    for i = 1:length(keypoints)
        if ~keep(i), continue; end
        kp1 = keypoints(i);
        r1 = kp1.y; c1 = kp1.x; s1 = kp1.scale; resp1 = kp1.response;
        scale_idx = find(filter_sizes == round(s1*9));
        % Check all scales, require smaller scales to be 200% stronger
        for j = 1:length(filter_sizes)
            if j == scale_idx, continue; end
            detH = detH_maps{j};
            half_size = floor(filter_sizes(j) / 2);
            r = round(r1); c = round(c1);
            if r < half_size+1 || r > height-half_size || c < half_size+1 || c > width-half_size
                continue;
            end
            if detH(r, c) > resp1 * 3.0 && filter_sizes(j) < filter_sizes(scale_idx)
                keep(i) = false;
                break;
            end
        end
    end
    keypoints = keypoints(keep);
end

% Helper: Remove Clustered Keypoints
function keypoints = removeClusteredKeypoints(keypoints, filter_sizes)
    keep = true(1, length(keypoints));
    for i = 1:length(keypoints)
        if ~keep(i), continue; end
        kp1 = keypoints(i);
        r1 = kp1.y; c1 = kp1.x; s1 = kp1.scale; resp1 = kp1.response;
        fs1 = round(s1 * 9);
        radius = 2 * fs1; % Scale-dependent radius
        for j = 1:length(keypoints)
            if i == j || ~keep(j), continue; end
            kp2 = keypoints(j);
            r2 = kp2.y; c2 = kp2.x; resp2 = kp2.response;
            dist = sqrt((r2-r1)^2 + (c2-c1)^2);
            if dist < radius && resp2 > resp1
                keep(i) = false;
                break;
            end
        end
    end
    keypoints = keypoints(keep);
end

% Helper: Compute Orientation
function ori = computeOrientation(int_img, x, y, scale, height, width)
    radius = round(6 * scale);
    responses_x = 0; responses_y = 0;
    for dy = -radius:radius
        for dx = -radius:radius
            if dx^2 + dy^2 > radius^2
                continue;
            end
            xx = round(x + dx); yy = round(y + dy);
            if xx < 1 || xx > width || yy < 1 || yy > height
                continue;
            end
            haar_size = round(2 * scale);
            wx = haarWavelet(int_img, xx, yy, haar_size, 0, height, width);
            wy = haarWavelet(int_img, xx, yy, haar_size, pi/2, height, width);
            weight = exp(-(dx^2 + dy^2) / (2 * (radius/2)^2));
            responses_x = responses_x + weight * wx;
            responses_y = responses_y + weight * wy;
        end
    end
    ori = atan2(responses_y, responses_x);
    if ori < 0
        ori = ori + 2*pi;
    end
end

% Helper: Haar Wavelet Response
function response = haarWavelet(int_img, x, y, size, angle, height, width)
    s = size / 2;
    x1 = round(x - s * cos(angle)); y1 = round(y - s * sin(angle));
    x2 = round(x + s * cos(angle)); y2 = round(y + s * sin(angle));
    half_s = round(s / 2);
    if angle == 0 % Horizontal
        pos = boxIntegral(int_img, x-half_s, y-half_s, x, y+half_s);
        neg = boxIntegral(int_img, x, y-half_s, x+half_s, y+half_s);
    else % Vertical
        pos = boxIntegral(int_img, x-half_s, y-half_s, x+half_s, y);
        neg = boxIntegral(int_img, x-half_s, y, x+half_s, y+half_s);
    end
    response = pos - neg;
end

% Helper: Compute 64D Descriptor
function desc = computeDescriptor(int_img, x, y, scale, orientation, height, width)
    desc = zeros(1, 64);
    grid_size = round(20 * scale);
    subregion = grid_size / 4;
    idx = 1;
    for i = -2:1:1
        for j = -2:1:1
            cx = x + i * subregion; cy = y + j * subregion;
            dx = 0; dy = 0; abs_dx = 0; abs_dy = 0;
            for di = -2:1:2
                for dj = -2:1:2
                    xx = round(cx + dj * scale); yy = round(cy + di * scale);
                    if xx < 1 || xx > width || yy < 1 || yy > height
                        continue;
                    end
                    haar_size = round(2 * scale);
                    wx = haarWavelet(int_img, xx, yy, haar_size, orientation, height, width);
                    wy = haarWavelet(int_img, xx, yy, haar_size, orientation + pi/2, height, width);
                    weight = exp(-((di*scale)^2 + (dj*scale)^2) / (2 * (subregion/2)^2));
                    dx = dx + weight * wx;
                    dy = dy + weight * wy;
                    abs_dx = abs_dx + weight * abs(wx);
                    abs_dy = abs_dy + weight * abs(wy);
                end
            end
            desc(idx:idx+3) = [dx, abs_dx, dy, abs_dy];
            idx = idx + 4;
        end
    end
    if norm(desc) > 0
        desc = desc / norm(desc);
    end
end