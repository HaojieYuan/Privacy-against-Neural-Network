function [scores_sort, index, maxlabel ]= predict(input_data, net)

scores = net.forward(input_data);
% probs = scores{1};
scores = mean(scores{1}, 2);
[scores_sort, index] = sort(scores,'descend');
maxlabel = index(1);

end