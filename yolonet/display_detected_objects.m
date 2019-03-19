function display_detected_objects(image, bound_boxes, class_labels, class_colormap)

% Show the image 
figure(); 
imshow(image);
hold on;

% Display bounding boxes around the detected objects
for i = 1:length(bound_boxes)
    if (bound_boxes(i).keep == 1)
        textstr = class_labels(bound_boxes(i).class_index);
        textcolor = class_colormap{bound_boxes(i).class_index};
        rectcolor = textcolor;

        title_coords = bound_boxes(i).coords-1;
        title_coords(2) = title_coords(2)-12;
        title_coords(3) = 40;
        title_coords(4) = 12;

        rectangle('Position', title_coords, 'EdgeColor', rectcolor, 'FaceColor', rectcolor);
        textpos = [title_coords(1)+4 title_coords(2)+4];
        text(textpos(1), textpos(2), textstr, 'Color', [0, 0, 0], 'fontWeight', 'bold', 'fontSize', 12);

        rectangle('Position', bound_boxes(i).coords, 'EdgeColor', rectcolor, 'LineWidth', 2);
    end
end

