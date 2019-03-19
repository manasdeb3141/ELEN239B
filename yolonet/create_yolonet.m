function yolonet=create_yolonet(mat_yolonet)

% Get the layer graph of the pre-trained MATLAB yolonet
lgraph = layerGraph(mat_yolonet.Layers);


% MATLAB's yolonet has a classifier as the output layer
% Replace the softmax+classifier layers with a regression
% layer
%
% Also, replace the last leakyRelu layer with a regular relu
% layer
lgraph = removeLayers(lgraph,'ClassificationLayer');
lgraph = removeLayers(lgraph,'softmax');
relu_layer = reluLayer('Name', 'relu_1');
regress_layer = regressionLayer('Name', 'output');
lgraph = addLayers(lgraph, regress_layer);
lgraph = replaceLayer(lgraph, 'leakyrelu_25', relu_layer);
lgraph = connectLayers(lgraph, 'FullyConnectedLayer1', 'output');
yolonet = assembleNetwork(lgraph);



