%function[] = filter_show(image)

image=imread('image.png');

prompt = 'enter an n by n odd filter matrix';
H1 = input(prompt);

filter_image = manual_filter(image, H1);
%%
figure
subplot(1,2,1)
imshow(image,[])
subplot(1,2,2)
imshow(filter_image,[])
%%
Y_image = abs(fft2(double(image)));
Y_filter_image = abs(fft2(double(filter_image)));
figure
imagesc(fftshift(log(Y_image+1))); colormap(gray);truesize;axis off;
figure
imagesc(fftshift(log(Y_filter_image+1))); colormap(gray);truesize;axis off;
H1f=abs(freqz2(H1)); 
figure1
imagesc((log(H1f+1)));colormap(gray);truesize;axis off;
%end