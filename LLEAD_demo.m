
%% reading the low light image to be enhanced
im = imread('8_2.png');
if(size(im,3)==3)
    im = rgb2gray(im);
end
im = double(im);  
[Ny,Nx] = size(im);

%% reading the dark channel prior enhanced image
im_dcp = imread('8_2_dcp.png');
if(size(im_dcp,3) == 3)
    im_dcp = rgb2gray(im_dcp);
end
im_dcp = double(im_dcp);

%% Noise Parameters
sig = estimate_noise(im);
PS = ones(size(im));	% power spectral density (in this case, flat, i.e., white noise)
seed = 0;               % random seed

%% Pyramidal representation parameters
Nsc = 4;    % 4 scales
Nor = 8;	% 8 orientations	

repres1 = 'fs';                    % steerable pyramid
repres2 = '';                      % dummy variable for steerable pyramimd

%% Model parameters 
blSize = [3 3];	    % n x n coefficient neighborhood of spatial neighbors within the same subband
                    % (n must be odd): 
parent = 0;	
                   
boundary = 1;		
covariance = 1;     
optim = 1;          

%% Call the denoising and enhancing function
im_d = denoise_enhance(im,im_dcp, sig, PS, blSize, parent, boundary, Nsc, Nor, covariance, optim, repres1);
im_d = real(im_d);
figure, imshow(uint8(im_d))