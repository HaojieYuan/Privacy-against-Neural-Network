function [input_data_extend, scores_steps] = trick(im, desied_output, net, n_steps)

input_data = {prepare_image(im)};
scores = net.forward(input_data);
scores = mean(scores{1}, 2);

scores_steps = single(zeros(size(scores,1), n_steps));
scores_steps(:,1) = scores;
input_data_extend = input_data{1};
yita = 0.9;

for ii = 2 : n_steps
    % calculate the gradient
    gradient = compute_gradient(net,desied_output);
    gradient_extend = gradient{1};    
%     input_data_extend = adveserial(yita/n_steps, input_data_extend, gradient_extend);
    delta = sign(gradient_extend);
    input_data_extend = input_data_extend + delta * yita / n_steps;
    scores = net.forward({input_data_extend});
    scores = mean(scores{1}, 2);
    scores_steps(:,ii) = scores;
end


end