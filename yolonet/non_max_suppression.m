function boxes_out=non_max_suppression(boxes_in, iou_thresh)

boxes_out = boxes_in;

for i=1:length(boxes_out)
    for j=i+1:length(boxes_out)
        % Calculate IOU = intersection / union
        intersect_area = rectint(boxes_out(i).coords, boxes_out(j).coords);
        box1_area = boxes_out(i).coords(3) * boxes_out(i).coords(4);
        box2_area = boxes_out(j).coords(3) * boxes_out(j).coords(4);
        union_area = box1_area + box2_area - intersect_area;

        IOU(i, j) = intersect_area / union_area;

        % If intersection over union of any two bounding boxes is higher than a threshold 
        % and the two boxes detected the same class, remove the box with the the lower box probability.
        if ((boxes_out(i).class_index == boxes_out(j).class_index) && (IOU(i, j) > iou_thresh))
            [box_prob drop_index] = min([boxes_out(i).box_prob, boxes_out(j).box_prob]);

            if (drop_index == 1)
                boxes_out(i).keep = 0;
            else
                boxes_out(j).keep = 0;
            end
        end
    end
end



