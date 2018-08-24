function[filter_image1] = manual_filter(image, filt1)
image = double(image);
[f1, f2] = size(filt1);
pad_image = padarray(image,[(f1-1)/2 (f2-1)/2]);
[x,y] = size(pad_image);

sum = 0;
for i = 1+((f1-1)/2):x-1%((f1-1)/2)
    for j = 1+((f2-1)/2):y - 1%((f2-1)/2)
        for k = 1:f1
            for l = 1:f2
                sum = sum + (pad_image(i-(1+((f1-1)/2))+k, j-(1+((f2-1)/2))+l)*filt1(k,l));
            end
        end
        filter_im(i-1,j-1) = sum;
        sum = 0;
              
    end
end

a=filter_im;
filter_image1 = padarray(filter_im,[(f1-1)/2 (f2-1)/2])
max_im = max(max(filter_image1))
min_im = min(min(filter_image1))
filter_image1 = round((filter_image1-min_im)*255/(max_im-min_im));
end