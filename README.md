classdef channelAttentionLayer < nnet.layer.Layer
    properties
        ReductionFactor 
    end
    
    properties (Learnable)
        Weights1 
        Weights2 
    end
    methods
        function layer = channelAttentionLayer(name, reductionFactor)
            layer.Name = name;
            layer.ReductionFactor = reductionFactor;
        end
        
        function layer = initialize(layer, layout)
            inputSize = layout.Size;
            
            if numel(inputSize) < 3
                error('error');
            end
            
            inputChannels = inputSize(3); 

            reduced_dim = max(1, floor(inputChannels / layer.ReductionFactor));
            layer.Weights1 = dlarray(randn(reduced_dim, inputChannels) * 0.01);
            layer.Weights2 = dlarray(randn(inputChannels, reduced_dim) * 0.01);
        end
        
        function Z = predict(layer, X)
            gap = mean(X, [1 2]);
            gap = reshape(gap, [], size(gap,4));
            weights = layer.Weights1 * gap; 
            weights = relu(weights);
            weights = layer.Weights2 * weights; 
            weights = sigmoid(weights);
            weights = reshape(weights, 1, 1, [], size(X,4)); % [1, 1, C, N]
            Z = X .* weights;
        end
    end
end
