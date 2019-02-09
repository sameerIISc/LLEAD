function fh = decomp_reconst_WU_mod_final(im,im_dcp,Nsc,Nor,block,noise,parent,covariance,optim,sig)

%% building the steerable pyramids for image and noise image
df = dtfilters('dddtf1');
DT = dddtree2('realdddt',im,3,df{1},df{2});
DTn = dddtree2('realdddt',noise,3,df{1},df{2});
DT_dcp = dddtree2('realdddt',im_dcp,3,df{1},df{2});
DT_enh = DT_dcp;

%% loading the coefficients and all
load('scale1_thtCoefs.mat');
thtc1 = gpuArray(x);
% thtc1 = [thtc1;0];
load('scale1_kCoefs.mat');
kc1 = gpuArray(x);
load('scale2_thtCoefs.mat');
thtc2 = gpuArray(x);
% thtc2 = [thtc2;0];
load('scale2_kCoefs.mat');
kc2 = gpuArray(x);
load('scale3_thtCoefs.mat');
thtc3 = gpuArray(x);
% thtc3 = [thtc3;0];
load('scale3_kCoefs.mat');
kc3 = gpuArray(x);    
load('dcpValues.mat');
dcp_values = x1;
prnt = 0;


%% computing the parameters for local prior for transmission map 
ti = 0.01:0.02:0.1;
summ_vec = 0.02*ones(size(ti));
temp = 0.11:0.04:0.4;
ti = [ti,temp];
summ_vec = [summ_vec, 0.04*ones(size(temp))];
temp = 0.41:0.08:1;
ti = [ti,temp];
summ_vec = [summ_vec, 0.08*ones(size(temp))];

ti = gpuArray((ti));
ti = ti(:);
t2_mtx = repmat(ti.^2,[1,9]);
t_mtx = repmat(ti,[1,9]);
nsamp_ti = length(ti);

for index = 1:1:48
    scale = floor((index-1)/16)+1;
    kc = (scale==1)*kc1 + (scale ==2)*kc2 + (scale==3)*kc3;
    thtc = (scale==1)*thtc1 + (scale ==2)*thtc2 + (scale==3)*thtc3;
   
    kx = floor(rem(index-1,16)/8)+1;
    dx = rem(index-1,8)+1;
    
    %% 1. extracting nth subband from low light and noise image
    y = gpuArray(DT.cfs{scale}(:,:,dx,kx));
    noise = gpuArray(DTn.cfs{scale}(:,:,dx,kx));
    sub_dcp = gpuArray(DT_dcp.cfs{scale}(:,:,dx,kx));
    T = abs(y)./(abs(sub_dcp) + 0.01);
    [Nsy,Nsx] = size(y);
    noise = noise*sqrt(((Nsy-2)*(Nsx-2))/(Nsy*Nsx));     % because we are discarding 2 coefficients on every dimension

    %% enhancing each band    
    Lx = (block(2)-1)/2;
    [nv,nh,~] = size(y);
    nblv = nv-block(1)+1;	% Discard the outer coefficients 
    nblh = nh-block(2)+1;   % for the reference (centrral) coefficients (to avoid boundary effects)
    nexp = nblv*nblh;			% number of coefficients considered
    N = 9; % size of the neighborhood
    Ly = (block(1)-1)/2;		% block(1) and block(2) m

    cent = floor((prod(block)+1)/2);	% reference coefficient in the neighborhood 

    %% 4. Compute covariance of noise from 'noise'
    Y = [];
    Tr = [];
    W = [];
    for j1 = 1:3
        for k1 = 1:3
            Y = cat(1,Y,reshape(y(k1:end-(3-k1),j1:end-(3-j1)),1,[]));
            Tr = cat(1,Tr,reshape(T(k1:end-(3-k1),j1:end-(3-j1)),1,[]));
            W = cat(1,W,reshape(noise(k1:end-(3-k1),j1:end-(3-j1)),1,[]));
        end
    end
    Y = Y';
    Tr = Tr';
    W = W';
    C_w = W'*W/nexp;
    
    %% 5. prior for the multiplier p(z) 
    lzmin = -20.5;
    lzmax = 3.5;
    step = 2;

    step1 = 2;
    lzi1 = lzmin:step1:lzmax;
    zi1 = exp(lzi1);

    lzi = lzmin:step:lzmax;
    nsamp_z = length(lzi);
    zi = exp(lzi);

    p_z = ones(nsamp_z,1);    % Flat log-prior (non-informative for GSM)
    p_z = p_z/sum(p_z);

    %% 7. computing second moment of t using local stats 
    T_vec = mean(Tr,2);
    T_vec(T_vec > 1) = 0.99;
    zi = zi(:);
    p_z = p_z(:);

%     temp = (k_tz(1:12).^2).*theta_tz(1:12) + (k_tz(1:12).*theta_tz(1:12)).^2;
    trm1 = sum(p_z(1:10)'.*zi(1:10)');

    [PT,X] = hist(T_vec,10);
    PT = PT./sum(PT);
    X = X(:);

    k_vec = [X.^8,X.^7,X.^6,X.^5,X.^4,X.^3,X.^2,X.^1,ones(size(X))]*kc(:);
    tht_vec = [X.^8,X.^7,X.^6,X.^5,X.^4,X.^3,X.^2,X.^1,ones(size(X))]*thtc(:);
    
        t2_T_vec = (k_vec.*(tht_vec.^2)) + ((k_vec.*tht_vec).^2);
    t2_T_vec = t2_T_vec(:);
    PT = PT(:);
    temp = sum(t2_T_vec.*PT);
    t2_avg = temp;

    %% 8. computing C_u 
    [S,dd] = eig(C_w);
    S = S*real(sqrt(dd));	
    iS = pinv(S);
    C_y = Y'*Y/nexp;
    C_x = C_y - C_w;
    C_x = C_x./t2_avg;
    [Q,L] = eig(C_x);
    
    %% 9. correct possible negative eigenvalues, without changing the overall variance
    L = diag(diag(L).*(diag(L)>0))*sum(diag(L))/(sum(diag(L).*(diag(L)>0))+(sum(diag(L).*(diag(L)>0))==0));
    C_x = Q*L*Q';
    [Q,L] = eig(iS*C_x*iS');	 % Double diagonalization of signal and noise
    la = diag(L);				 % eigenvalues: energy in the new represetnation.
    la = real(la).*(real(la)>0);
    la = la';

    %% 10. Linearly transform the observations, and keep the quadratic values 
    V = Q'*iS*Y';
    V2 = V.^2;
    M = S*Q;
    m = real(M(cent,:));

    %% 11. prior for the multiplier p(z)
    lzmin = -20.5;
    lzmax = 3.5;
    step = 2;

    lzi = lzmin:step:lzmax;
    nsamp_z = length(lzi);
    zi = exp(lzi);

    p_z = ones(nsamp_z,1);    % Flat log-prior (non-informative for GSM)
    p_z = p_z/sum(p_z);

    %% 12. some matrices for computations
    lad = diag(la);
    mlad = diag(m(:).*la(:));
        
    tz = 11;
    tz1 = 3;
    
    %% 13. computing the local prior for transmission map 
    k = [T_vec.^8,T_vec.^7,T_vec.^6,T_vec.^5,T_vec.^4,T_vec.^3,T_vec.^2,T_vec.^1,ones(size(T_vec))]*kc(:);
    theta = [T_vec.^8,T_vec.^7,T_vec.^6,T_vec.^5,T_vec.^4,T_vec.^3,T_vec.^2,T_vec.^1,ones(size(T_vec))]*thtc(:);

    k = k(:)';
    theta = theta(:)';
    theta(theta <= 0) = 0.01;
    
    
    k = gpuArray(single(k));
    theta = gpuArray(single(theta));
    gt = (1./(gamma(k).*(theta.^k)));
    ti_rep = repmat(ti,1,length(k));
       

    p_t = gt.*(ti_rep.^(k-1)).*(exp(-ti_rep./theta));
    p_t = p_t.*repmat(summ_vec(:),[1,nexp]);
    p_t = p_t./repmat(sum(p_t,1),[nsamp_ti,1]);

%%  14. For z less than tau 
    zi = gpuArray(zi);
    p_t_zT = p_t;
    p_z = gpuArray(single(p_z));
    V2 = single(V2);
    V = single(V);

    est = gpuArray.zeros(13,nexp);
    den = gpuArray.zeros(13,nexp);

    p_z1 = p_z(:)';
    laz = la(:)*zi(1:tz-1);
    pg1_lz = 1./sqrt(prod(1+laz,1));
    aux = exp(-0.5*V2'*(1./1+laz));
    p_y_zt1 = aux*diag(pg1_lz);
    dist = p_y_zt1.*repmat(p_z1(1:tz-1),[nexp,1]);
    aux = diag(m)*(laz./(1+laz));
    mu_x = V'*aux;
    est(1:tz-1,:) = (mu_x.*dist)';
    den(1:tz-1,:) = dist';
    
 %% 15. loop version of integration over t 
    zi = zi';
    aux1 = (t2_mtx*lad);
    mlad = real(mlad);
    aux2 = (t_mtx*mlad);
    zi3 = reshape(zi(tz:13),[1,1,tz1]);
    aux13 = repmat(aux1,[1,1,tz1]);
    aux13 = aux13 + 1; 
    aux23 = repmat(aux2,[1,1,tz1]);
    aux33 = pagefun(@times,aux13,zi3);
    aux43 = pagefun(@times,aux23,zi3);
    temp23 = pagefun(@rdivide,1,sqrt(prod(aux33,2)));

    for i = tz:13
        aux3 = aux33(:,:,i-(tz-1));
        aux4 = aux43(:,:,i-(tz-1));
        temp2 = temp23(:,:,i-(tz-1));
        temp1 = exp(-0.5*(1./aux3)*V2);
        p_y_zt = bsxfun(@times,temp1,temp2);
        int_part = sum(bsxfun(@times,p_y_zt,p_t_zT),1);
        den(i,:) = int_part*p_z(i);
        temp = ((aux4./aux3)*V).*p_t_zT;
        est(i,:) = sum(temp,1).*(den(i,:));
    end


%% 17.
    est = sum(est,1)./sum(den,1);
    x_hat = sub_dcp(:,:,1);
    x_hat(1+Ly:nblv+Ly,1+Lx:nblh+Lx) = real(reshape(est(:),nblv,nblh));
    x_hat(isnan(x_hat)) = sub_dcp(isnan(x_hat));
    x_hat = gather(x_hat);
%% putting the enhanced band back
    DT_enh.cfs{scale}(:,:,dx,kx) = x_hat;
end
fh = idddtree2(DT_enh);
end