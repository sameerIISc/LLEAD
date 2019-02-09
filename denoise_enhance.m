function im_d = denoi_enhance_BLS_GSM(im, im_dcp, sig, PS, blSize, parent, boundary, Nsc, Nor, covariance, optim, repres1)

im_dcp = double(im_dcp);

if ~exist('PS'),
    no_white = 0;   % Power spectral density of noise. Default is white noise
else
	no_white = 1;
end

blSzX = blSize(1);
blSzY = blSize(2);
[Ny Nx] = size(im);


%% We ensure that the processed image has dimensions that are integer
Npy = ceil(Ny/2^(Nsc+1))*2^(Nsc+1);
Npx = ceil(Nx/2^(Nsc+1))*2^(Nsc+1);

if Npy~=Ny || Npx~=Nx,
    Bpy = Npy-Ny;
    Bpx = Npx-Nx;
    im = bound_extension(im,Bpy,Bpx,'mirror');
    im_dcp = bound_extension(im_dcp,Bpy,Bpx,'mirror');
    im = im(Bpy+1:end,Bpx+1:end);	% add stripes only up and right
    im_dcp = im_dcp(Bpy+1:end,Bpx+1:end);
end   


%% size of the extension for boundary handling
if (repres1 == 's') | (repres1 == 'fs'),
    By = (blSzY-1)*2^(Nsc-2);
    Bx = (blSzX-1)*2^(Nsc-2);
else        
    By = (blSzY-1)*2^(Nsc-1);
    Bx = (blSzX-1)*2^(Nsc-1);
end    

if ~no_white,       % White noise
    PS = ones(size(im));
end

%% As the dimensions of the power spectral density (PS) support and that of the
PS = fftshift(PS);
isoddPS_y = (size(PS,1)~=2*(floor(size(PS,1)/2)));
isoddPS_x = (size(PS,2)~=2*(floor(size(PS,2)/2)));
PS = PS(1:end-isoddPS_y, 1:end-isoddPS_x);          % ensures even dimensions for the power spectrum
PS = fftshift(PS);

[Ndy,Ndx] = size(PS);   % dimensions are even

delta = real(ifft2(sqrt(PS)));      
delta = fftshift(delta);
aux = delta;
delta = zeros(Npy,Npx);
if (Ndy<=Npy)&&(Ndx<=Npx),
    delta(Npy/2+1-Ndy/2:Npy/2+Ndy/2,Npx/2+1-Ndx/2:Npx/2+Ndx/2) = aux;
elseif (Ndy>Npy)&&(Ndx>Npx),   
    delta = aux(Ndy/2+1-Npy/2:Ndy/2+Npy/2,Ndx/2+1-Npx/2:Ndx/2+Npx/2);
elseif (Ndy<=Npy)&&(Ndx>Npx),   
    delta(Npy/2+1-Ndy/2:Npy/2+1+Ndy/2-1,:) = aux(:,Ndx/2+1-Npx/2:Ndx/2+Npx/2);
elseif (Ndy>Npy)&&(Ndx<=Npx),   
    delta(:,Npx/2+1-Ndx/2:Npx/2+1+Ndx/2-1) = aux(Ndy/2+1-Npy/2:Ndy/2+1+Npy/2-1,:);
end 

%% Boundary handling: it extends im and delta
if boundary,
    im = bound_extension(im,By,Bx,'mirror');
    im_dcp = bound_extension(im_dcp,By,Bx,'mirror');
    if repres1 == 'w',
        delta = bound_extension(delta,By,Bx,'mirror');
    else    
        aux = delta;
        delta = zeros(Npy + 2*By, Npx + 2*Bx);
        delta(By+1:By+Npy,Bx+1:Bx+Npx) = aux;
    end    
else
	By=0;Bx=0;
end

delta = delta/sqrt(mean2(delta.^2));    % Normalize the energy (the noise variance is given by "sig")
delta = sig*delta;                      % Impose the desired variance to the noise

%% main
im_d = decomp_reconst_WU(im,im_dcp,Nsc, Nor, [blSzX blSzY], delta, parent, covariance, optim, sig);        
im_d = im_d(By+1:By+Npy,Bx+1:Bx+Npx);   
im_d = im_d(1:Ny,1:Nx);