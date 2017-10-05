function gradient = compute_gradient(net, desied_output)

% get an empty set of probablities
probs = single(zeros(net.blobs('prob').shape));
% set the probablity of our intended outcome to 1
probs(desied_output,:) = 1;
% do backpropagation to calculate the gradient for that outcome 
gradient = net.backward({probs});


end