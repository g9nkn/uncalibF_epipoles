function md = distort_pts_k1(mu, k, c)

    if nargin < 3, c = zeros(2,1); end
    
    xu = mu(1,:) - c(1);
    yu = mu(2,:) - c(2);
   
    % distorted points directly
    r = 2*k*( xu.^2 + yu.^2 );
    rr = ( 1-sqrt(1-2*r) ) ./ r;
    xd = xu .* rr;
    yd = yu .* rr;
    
    md = [xd + c(1); 
          yd + c(2)];
    
    if size(mu,1)==3
        md(3,:) = 1;
    end
    
end