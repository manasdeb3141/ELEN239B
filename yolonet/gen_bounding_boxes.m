function bound_boxes = gen_bounding_boxes(data_cube, box_prob, box_index, detected_cells, class_index)

% Smaller data cube contains only the box coords 
box_coord_data_cube = data_cube(:, :, 23:30);

count = 0;

for i=1:7
    for j=1:7
        if (detected_cells(i, j) == 1)
            % An obect was detected in this cell
            count = count + 1;

            % Get the center of the bounding box relative to the cell
            x = box_coord_data_cube(i, j, 1+((box_index(i, j)-1)*4));
            y = box_coord_data_cube(i, j, 2+((box_index(i, j)-1)*4));


            % Get the width and height of the bounding boxes relative to the image
            % YOLO outputs the square root of the width and height
            w = box_coord_data_cube(i, j, 3+((box_index(i, j)-1)*4))^2;
            h = box_coord_data_cube(i, j, 4+((box_index(i, j)-1)*4))^2;

            % Scale to image size
            w_scaled = w * 448;
            h_scaled = h * 448;
            x_scaled = ((j-1)*448/7) + (x*448/7) - (w_scaled/2);
            y_scaled = ((i-1)*448/7) + (y*448/7) - (h_scaled/2);

            % Store the co-ordinates of the bounding box
            bound_boxes(count).coords = [x_scaled, y_scaled, w_scaled, h_scaled];

            % Store the cell index of the box
            bound_boxes(count).cell_index = [i, j];

            % Store the clas index
            bound_boxes(count).class_index = class_index(i, j);

            % Store the box probability
            bound_boxes(count).box_prob = box_prob(i, j);

            % Flag to keep the box (default)
            bound_boxes(count).keep = 1;
        end
    end
end


