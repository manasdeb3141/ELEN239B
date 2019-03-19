function runyolo(img_fname)

if ~exist('img_fname', 'var')
    img_fname='pedestrian1.jpg';
end

% Set defaults for the class probability and IOU thresholds
if ~exist('prob_thresh', 'var')
    prob_thresh = 0.10;
end

if ~exist('iou_thresh', 'var')
    iou_thresh = 0.4;
end

% MATLAB's yolonet was trained on PASCAL VOC dataset which comprises of images of the following 20 classes
class_labels = {'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', ...
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', ...
               'sheep', 'sofa', 'train', 'tvmonitor'};

class_colormap = {[   0.5,    0.5,    0.5], ...          % Gray
                  [0.1836, 0.3086, 0.3086], ...          % DarkSlateGray
                  [     1,      0,      0], ...          % Red
                  [     1, 0.7500, 0.7930], ...          % Pink
                  [     1, 0.0781, 0.5742], ...          % DeepPink
                  [     1, 0.6445,      0], ...          % Orange
                  [     1, 0.2695,      0], ...          % OrangeRed
                  [     1,      1,      0], ...          % Yellow
                  [0.7383, 0.7148, 0.4180], ...          % DarkKhaki
                  [0.6445, 0.1641, 0.1641], ...          % Brown
                  [0.9542, 0.6406, 0.3750], ...          % SandyBrown
                  [0.7188, 0.5234, 0.0430], ...          % DarkGoldenRod
                  [0.8203, 0.4102, 0.1172], ...          % Chocolate
                  [     0, 0.5000, 0.0000], ...          % Green
                  [0.6758, 1.0000, 0.1836], ...          % GreenYellow
                  [0.2344, 0.6992, 0.4414], ...          % MediumSeaGreen
                  [0.0000, 0.5000, 0.5000], ...          % Teal
                  [0.0000, 0.0000, 1.0000], ...          % Blue
                  [0.0000, 1.0000, 1.0000], ...          % Cyan
                  [0.5000, 0.0000, 0.5000]};             % Purple


% Form the full path to the image file name
datadir='data\';
img_fname_full=strcat(datadir, img_fname);

% If the YOLO neural network was saved to a file, then load it from there
% else create it
if exist('newyolonet.mat', 'file')
    disp('Loading previously saved YOLONET');
    load('newyolonet.mat');
else
    disp('Creating YOLONET');

    % Use the weights from the pre-trained MATLAB YOLO net
    if exist('yolonet.mat', 'file')
        data=load('yolonet.mat');
    else
        error('Unable to find yolonet.mat');
    end

    % Create the YOLO neural network and save it to a file
    newyolonet = create_yolonet(data.yolonet);
    save('newyolonet.mat', 'newyolonet');
end

% Load the image file and display the image
image = single(imresize(imread(img_fname_full),[448 448]))/255;
figure(1);
imagesc(image);
title('Image file input to YOLO network');

% Input the image to the YOLO neural network and get a prediction of
% the image class probabilities and the bounding box dimensions and
% probabilities per cell
regress_out = predict(newyolonet, image, 'ExecutionEnvironment', 'gpu');

% Form a data cube of 7x7x30 from the regressor's output
data_cube = create_datacube(regress_out);

% For each cell, find out which of the 2 boxes has the highest probability
%  box_idx == 1 => the vertical box has higher probability
%  box_idx == 2 => the horizontal box has higher probability
[box_prob, box_index] = max(data_cube(:, :, 21:22), [], 3);
detected_cells = box_prob > prob_thresh;

% For each cell determine the object class that has the highest probability
% The class with the highest probability is stored in class_index
[class_prob, class_index] = max(data_cube(:, :, 1:20), [], 3);

% Show the cells in the 7x7 grid where objects were detected
figure(2);
imagesc(detected_cells);
title('Cells where objects were detected');

% Generate the bounding boxes
bound_boxes = gen_bounding_boxes(data_cube, box_prob, box_index, detected_cells, class_index);
display_detected_objects(image, bound_boxes, class_labels, class_colormap);
title('Dtected objects Without Non-Maximal Suppression');

% Perform the non-max suppression on these bounding boxes
bound_boxes = non_max_suppression(bound_boxes, iou_thresh);
display_detected_objects(image, bound_boxes, class_labels, class_colormap);
title('Detected objects with Non-Maximal Suppression');



