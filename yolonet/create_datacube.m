function data_cube=create_datacube(nn_output)


class_prob = reshape(nn_output(1:980), 20, []);
box_prob = reshape(nn_output(981:1078), 2, []);
box_dims = reshape(nn_output(1079:1470), 8, []);

data_cube=zeros(7, 7, 30);

cell_idx = 1;
for i=1:7
    for j=1:7
        data_cube(i, j, :) = [class_prob(:, cell_idx); box_prob(:, cell_idx); box_dims(:, cell_idx)];
        cell_idx = cell_idx + 1;
    end
end



