function lgraph=create_yolonet_layers()


lgraph = layerGraph();

layers = [
    imageInputLayer([448 448 3], 'Name', 'ImageInputLayer', 'Normalization', 'none') 

    convolution2dLayer(7, 64, 'NumChannels', 3, 'Padding', [3 3 3 3], 'Stride', [2 2], 'Name', 'Convolution2DLayer')
    leakyReluLayer(0.1, 'Name','leakyrelu_1')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'MaxPooling2DLayer0')

    convolution2dLayer(3, 192, 'NumChannels', 64, 'Padding', [1 1 1 1], 'Stride', [1 1], 'Name', 'Convolution2DLayer0')
    leakyReluLayer(0.1, 'Name','leakyrelu_2')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'MaxPooling2DLayer1')

    convolution2dLayer(1, 128, 'NumChannels', 192, 'Padding', [0 0 0 0], 'Stride', [1 1], 'Name', 'Convolution2DLayer1')
    leakyReluLayer(0.1, 'Name','leakyrelu_3')

    convolution2dLayer(3, 256, 'NumChannels', 128, 'Padding', [1 1 1 1], 'Stride', [1 1], 'Name', 'Convolution2DLayer2')
    leakyReluLayer(0.1, 'Name','leakyrelu_4')

    convolution2dLayer(1, 256, 'NumChannels', 256, 'Padding', [0 0 0 0], 'Stride', [1 1], 'Name', 'Convolution2DLayer3')
    leakyReluLayer(0.1, 'Name','leakyrelu_5')

    convolution2dLayer(3, 512, 'NumChannels', 256, 'Padding', [1 1 1 1], 'Stride', [1 1], 'Name', 'Convolution2DLayer4')
    leakyReluLayer(0.1, 'Name','leakyrelu_6')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'MaxPooling2DLayer2')

    convolution2dLayer(1, 256, 'NumChannels', 512, 'Padding', [0 0 0 0], 'Stride', [1 1], 'Name', 'Convolution2DLayer5')
    leakyReluLayer(0.1, 'Name','leakyrelu_7')

    convolution2dLayer(3, 512, 'NumChannels', 256, 'Padding', [1 1 1 1], 'Stride', [1 1], 'Name', 'Convolution2DLayer6')
    leakyReluLayer(0.1, 'Name','leakyrelu_8')

    convolution2dLayer(1, 256, 'NumChannels', 512, 'Padding', [0 0 0 0], 'Stride', [1 1], 'Name', 'Convolution2DLayer7')
    leakyReluLayer(0.1, 'Name','leakyrelu_9')

    convolution2dLayer(3, 512, 'NumChannels', 256, 'Padding', [1 1 1 1], 'Stride', [1 1], 'Name', 'Convolution2DLayer8')
    leakyReluLayer(0.1, 'Name','leakyrelu_10')

    convolution2dLayer(1, 256, 'NumChannels', 512, 'Padding', [0 0 0 0], 'Stride', [1 1], 'Name', 'Convolution2DLayer9')
    leakyReluLayer(0.1, 'Name','leakyrelu_11')

    convolution2dLayer(3, 512, 'NumChannels', 256, 'Padding', [1 1 1 1], 'Stride', [1 1], 'Name', 'Convolution2DLayer10')
    leakyReluLayer(0.1, 'Name','leakyrelu_12')

    convolution2dLayer(1, 256, 'NumChannels', 512, 'Padding', [0 0 0 0], 'Stride', [1 1], 'Name', 'Convolution2DLayer11')
    leakyReluLayer(0.1, 'Name','leakyrelu_13')

    convolution2dLayer(3, 512, 'NumChannels', 256, 'Padding', [1 1 1 1], 'Stride', [1 1], 'Name', 'Convolution2DLayer12')
    leakyReluLayer(0.1, 'Name','leakyrelu_14')

    convolution2dLayer(1, 512, 'NumChannels', 512, 'Padding', [0 0 0 0], 'Stride', [1 1], 'Name', 'Convolution2DLayer13')
    leakyReluLayer(0.1, 'Name','leakyrelu_15')

    convolution2dLayer(3, 1024, 'NumChannels', 512, 'Padding', [1 1 1 1], 'Stride', [1 1], 'Name', 'Convolution2DLayer14')
    leakyReluLayer(0.1, 'Name','leakyrelu_16')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'MaxPooling2DLayer3')

    convolution2dLayer(1, 512, 'NumChannels', 1024, 'Padding', [0 0 0 0], 'Stride', [1 1], 'Name', 'Convolution2DLayer15')
    leakyReluLayer(0.1, 'Name','leakyrelu_17')

    convolution2dLayer(3, 1024, 'NumChannels', 512, 'Padding', [1 1 1 1], 'Stride', [1 1], 'Name', 'Convolution2DLayer16')
    leakyReluLayer(0.1, 'Name','leakyrelu_18')

    convolution2dLayer(1, 512, 'NumChannels', 1024, 'Padding', [0 0 0 0], 'Stride', [1 1], 'Name', 'Convolution2DLayer17')
    leakyReluLayer(0.1, 'Name','leakyrelu_19')

    convolution2dLayer(3, 1024, 'NumChannels', 512, 'Padding', [1 1 1 1], 'Stride', [1 1], 'Name', 'Convolution2DLayer18')
    leakyReluLayer(0.1, 'Name','leakyrelu_20')

    convolution2dLayer(3, 1024, 'NumChannels', 1024, 'Padding', [1 1 1 1], 'Stride', [1 1], 'Name', 'Convolution2DLayer19')
    leakyReluLayer(0.1, 'Name','leakyrelu_21')

    convolution2dLayer(3, 1024, 'NumChannels', 1024, 'Padding', [1 1 1 1], 'Stride', [2 2], 'Name', 'Convolution2DLayer20')
    leakyReluLayer(0.1, 'Name','leakyrelu_22')

    convolution2dLayer(3, 1024, 'NumChannels', 1024, 'Padding', [1 1 1 1], 'Stride', [1 1], 'Name', 'Convolution2DLayer21')
    leakyReluLayer(0.1, 'Name','leakyrelu_23')

    convolution2dLayer(3, 1024, 'NumChannels', 1024, 'Padding', [1 1 1 1], 'Stride', [1 1], 'Name', 'Convolution2DLayer22')
    leakyReluLayer(0.1, 'Name','leakyrelu_24')

    fullyConnectedLayer(4096, 'Name', 'FullyConnectedLayer')
    leakyReluLayer(0.1, 'Name','leakyrelu_25')
    fullyConnectedLayer(1470, 'Name', 'FullyConnectedLayer1')

    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'ClassificationLayer')
    ];

lgraph = addLayers(lgraph, layers);

%figure;
%plot(lgraph);

