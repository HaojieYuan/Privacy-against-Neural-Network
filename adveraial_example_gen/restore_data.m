function im_data = restore_data(crops_data)

IMAGE_DIM = 256;
CROPPED_DIM = 224;
indices = [0 IMAGE_DIM-CROPPED_DIM] + 1;
center = floor(indices(2) / 2) + 1;

im_data(center:center+CROPPED_DIM-1,center:center+CROPPED_DIM-1,:) = crops_data(:,:,:,5);

n = 1;
for i = indices
  for j = indices
    im_data(i:i+CROPPED_DIM-1, j:j+CROPPED_DIM-1, :) = crops_data(:, :, :, n);    
    n = n + 1;
  end
end

d = load('../+caffe/imagenet/ilsvrc_2012_mean.mat');
mean_data = d.mean_data;
im_data = im_data + mean_data;
im_data = permute(im_data, [2, 1, 3]);
im_data = im_data(:, :, [3, 2, 1]); 

end