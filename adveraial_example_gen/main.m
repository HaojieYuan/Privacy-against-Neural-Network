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
% ��ԭʼͼ�����resize��crop and flip ����10��Views, ѵ��������ȡƽ��
input_data = {prepare_image(im)};
 %CNNԤ�������
[scores_sort, index, maxlabel]= predict(input_data, net);
% ϣ����ԭʼͼ����ֳɵ���
desied_output = 271;
% ѭ������100��
n_steps = 100;
% ��ʼ��ƭ
[input_data_extend, scores_steps] = trick(im, desied_output, net, n_steps);

%% illustrate the data
% ���ÿ�η�����
figure(1);
x = plot(1:n_steps, scores_steps(maxlabel,:), 'b-');
legend(x,'panda');
hold on;
y = plot(1:n_steps, scores_steps(desied_output,:), 'r-');
legend(y,'wolf');
xlabel('Iteration Number');
ylabel('Probablity');

%% restore data
% ���ɶԿ�����չʾ
stego_im_data = restore_data(input_data_extend);
% ԭʼͼƬչʾ
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
