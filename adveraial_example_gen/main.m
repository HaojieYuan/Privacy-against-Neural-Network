clear;clc;
close all;
clc;
clear mex;
clear is_valid_handle; % to clear init_key
% initialize the network
net = initial();

% prepare oversampled input
% input_data is Height x Width x Channel x Num
im = imread('../../examples/images/upload.jpg');
% 对原始图像进行resize、crop and flip 生成10个Views, 训练网络结果取平均
input_data = {prepare_image(im)};
 %CNN预测分类结果
[scores_sort, index, maxlabel]= predict(input_data, net);
% 希望将原始图像误分成的类
desied_output = 271;
% 循环迭代100次
n_steps = 100;
% 开始欺骗
[input_data_extend, scores_steps] = trick(im, desied_output, net, n_steps);

%% illustrate the data
% 输出每次分类结果
figure(1);
x = plot(1:n_steps, scores_steps(maxlabel,:), 'b-');
legend(x,'panda');
hold on;
y = plot(1:n_steps, scores_steps(desied_output,:), 'r-');
legend(y,'wolf');
xlabel('Iteration Number');
ylabel('Probablity');

%% restore data
% 生成对抗样本展示
stego_im_data = restore_data(input_data_extend);
% 原始图片展示
cover_im_data = restore_data(input_data{1});
figure(2);imshow(uint8(cover_im_data));title('cover image');
figure(3);imshow(uint8(stego_im_data));title('stego image');

%%
% input_data = {prepare_image(im)};
% [scores_sort, index, maxlabel ]= predict(input_data, net);
% 
% % calculate the gradient
% gradient = compute_gradient(net);
% input_data_extend = input_data{1};
% gradient_extend = gradient{1};

%% 20160313 old
% yita = 0.9;
% delta = sign(gradient_extend);
% input_data_extend = input_data_extend + delta * yita;
% [scores_sort_wish, index_wish, maxlabel_wish ]= predict({input_data_extend}, net);
