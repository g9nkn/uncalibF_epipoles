% Larsson's 6-point method for F+f estimation
%
% USAGE:
%   [F, f, E] = Ff6pt_Larsson(m1, m2, c1, c2)
% INPUTS:
%   m1,m2 - 3x6, 2D point correspondences in homogeneous coordinates 
%   c1,c2 - 1x2, image center of each camera
% OUTPUTS:
%   F - 3x3xM, fundamental matrix
%   f - 1xM, focal length
%   E - 3x3xM, essential matrix
%   (0 <= M <= 10)
% REFERENCE:
%   V. Larsson et al., "Efficient Solvers for Minimal Problems by Syzygy-based Reduction," CVPR 2017.  
function [F, focal, E] = Ff6pt_Larsson(x1, x2, center1, center2)

    narginchk(4,4);
    if size(x1,1) == 2, x1(3,:) = 1; end
    if size(x2,1) == 2, x2(3,:) = 1; end

    
    % normalization
    s = max(abs([x1(:);x2(:)]));
    T1 = diag([1/s,1/s,1]) * ...
         [1 0 -center1(1)
          0 1 -center1(2)
          0 0         1];
    T2 = diag([1/s,1/s,1]) * ...
         [1 0 -center2(1)
          0 1 -center2(2)
          0 0         1];
    x1 = T1*x1;
    x2 = T2*x2;

    
 
    % nullspace
    A = [x2(1,:)'.*x1',  x2(2,:)'.*x1', x2(3,:)'.*x1'];
    [~,~,V] = svd(A);
    data = [V(:,7); V(:,8); V(:,9)];    
    
    % solve 6pt focal-F problem
    sol = solver_relpose_6pt_focal(data);
    ind = prod(~imag(sol) & sol(3,:)>0, 1, 'native');
    sol = sol(:,ind);
    
    
    % recover solutions
    F  = [];
    F0 = reshape(V(:,7),3,3)';
    F1 = reshape(V(:,8),3,3)';
    F2 = reshape(V(:,9),3,3)';   
   

    % denormalization
    ind = false(1,size(sol,2));
    for i = 1:size(sol,2)
        tmp_F = F0 + sol(1,i)*F1 + sol(2,i)*F2;
        if check_chirality_F(tmp_F, x1, x2)
            tmp_F  = T2' * tmp_F * T1;
            tmp_F  = tmp_F / norm(tmp_F,'fro');
            F      = cat(3, F, tmp_F );
            ind(i) = true;
        end
    end
    focal = 1 ./ sqrt(sol(3,ind)) * s;

    if nargout > 2
         E  = zeros(3,3,length(focal));
         for i=1:length(focal)
            K1 = [focal(i)       0 center1(1)
                        0 focal(i) center1(2)
                        0        0         1];
            K2 = [focal(i)       0 center1(1)
                        0 focal(i) center1(2)
                        0        0         1];
            
            E_tmp = K2' * F(:,:,i) * K1;
            E(:,:,i) = E_tmp / norm(E_tmp,'fro');
         end
    end
    
end

function sols = solver_relpose_6pt_focal(data)
[C0,C1] = setup_elimination_template(data);
C1 = C0 \ C1;
RR = [-C1(end-7:end,:);eye(15)];
AM_ind = [16,12,1,2,3,13,4,17,5,6,18,7,19,20,8];
AM = RR(AM_ind,:);
[V,D] = eig(AM);
V = V ./ (ones(size(V,1),1)*V(1,:));
sols(1,:) = V(2,:);
sols(2,:) = diag(D).';
sols(3,:) = V(13,:);
end
% Action =  y
% Quotient ring basis (V) = 1,x,x^2,x*y,x*y*z,x*z,x*z^2,y,y^2,y^2*z,y*z,y*z^2,z,z^2,z^3,
% Available monomials (RR*V) = x^2*y,x*y^2,x*y^2*z,x*y*z^2,y^3,y^3*z,y^2*z^2,y*z^3,1,x,x^2,x*y,x*y*z,x*z,x*z^2,y,y^2,y^2*z,y*z,y*z^2,z,z^2,z^3,
function [coeffs] = compute_coeffs(data)
coeffs = zeros(280,1);
coeffs(1) = 2*data(12)*data(16)*data(18) - data(10)*data(18)^2;
coeffs(2) = -data(18)^2*data(19) + 2*data(16)*data(18)*data(21) + 2*data(12)*data(18)*data(25) + 2*data(12)*data(16)*data(27) - 2*data(10)*data(18)*data(27);
coeffs(3) = 2*data(18)*data(21)*data(25) - 2*data(18)*data(19)*data(27) + 2*data(16)*data(21)*data(27) + 2*data(12)*data(25)*data(27) - data(10)*data(27)^2;
coeffs(4) = 2*data(21)*data(25)*data(27) - data(19)*data(27)^2;
coeffs(5) = data(10)*data(12)^2 + 2*data(12)*data(13)*data(15) - data(10)*data(15)^2 + data(10)*data(16)^2 + 2*data(11)*data(16)*data(17) - data(10)*data(17)^2;
coeffs(6) = data(12)^2*data(19) - data(15)^2*data(19) + data(16)^2*data(19) - data(17)^2*data(19) + 2*data(16)*data(17)*data(20) + 2*data(10)*data(12)*data(21) + 2*data(13)*data(15)*data(21) + 2*data(12)*data(15)*data(22) + 2*data(12)*data(13)*data(24) - 2*data(10)*data(15)*data(24) + 2*data(10)*data(16)*data(25) + 2*data(11)*data(17)*data(25) + 2*data(11)*data(16)*data(26) - 2*data(10)*data(17)*data(26);
coeffs(7) = 2*data(12)*data(19)*data(21) + data(10)*data(21)^2 + 2*data(15)*data(21)*data(22) - 2*data(15)*data(19)*data(24) + 2*data(13)*data(21)*data(24) + 2*data(12)*data(22)*data(24) - data(10)*data(24)^2 + 2*data(16)*data(19)*data(25) + 2*data(17)*data(20)*data(25) + data(10)*data(25)^2 - 2*data(17)*data(19)*data(26) + 2*data(16)*data(20)*data(26) + 2*data(11)*data(25)*data(26) - data(10)*data(26)^2;
coeffs(8) = data(19)*data(21)^2 + 2*data(21)*data(22)*data(24) - data(19)*data(24)^2 + data(19)*data(25)^2 + 2*data(20)*data(25)*data(26) - data(19)*data(26)^2;
coeffs(9) = 2*data(9)*data(12)*data(16) - 2*data(9)*data(10)*data(18) + 2*data(7)*data(12)*data(18) + 2*data(3)*data(16)*data(18) - data(1)*data(18)^2;
coeffs(10) = -2*data(9)*data(18)*data(19) + 2*data(9)*data(16)*data(21) + 2*data(7)*data(18)*data(21) + 2*data(9)*data(12)*data(25) + 2*data(3)*data(18)*data(25) - 2*data(9)*data(10)*data(27) + 2*data(7)*data(12)*data(27) + 2*data(3)*data(16)*data(27) - 2*data(1)*data(18)*data(27);
coeffs(11) = 2*data(9)*data(21)*data(25) - 2*data(9)*data(19)*data(27) + 2*data(7)*data(21)*data(27) + 2*data(3)*data(25)*data(27) - data(1)*data(27)^2;
coeffs(12) = data(10)^3 + data(10)*data(11)^2 + data(10)*data(13)^2 + 2*data(11)*data(13)*data(14) - data(10)*data(14)^2;
coeffs(13) = 3*data(10)^2*data(19) + data(11)^2*data(19) + data(13)^2*data(19) - data(14)^2*data(19) + 2*data(10)*data(11)*data(20) + 2*data(13)*data(14)*data(20) + 2*data(10)*data(13)*data(22) + 2*data(11)*data(14)*data(22) + 2*data(11)*data(13)*data(23) - 2*data(10)*data(14)*data(23);
coeffs(14) = 3*data(10)*data(19)^2 + 2*data(11)*data(19)*data(20) + data(10)*data(20)^2 + 2*data(13)*data(19)*data(22) + 2*data(14)*data(20)*data(22) + data(10)*data(22)^2 - 2*data(14)*data(19)*data(23) + 2*data(13)*data(20)*data(23) + 2*data(11)*data(22)*data(23) - data(10)*data(23)^2;
coeffs(15) = data(19)^3 + data(19)*data(20)^2 + data(19)*data(22)^2 + 2*data(20)*data(22)*data(23) - data(19)*data(23)^2;
coeffs(16) = 2*data(3)*data(10)*data(12) + data(1)*data(12)^2 + 2*data(6)*data(12)*data(13) - 2*data(6)*data(10)*data(15) + 2*data(4)*data(12)*data(15) + 2*data(3)*data(13)*data(15) - data(1)*data(15)^2 + 2*data(7)*data(10)*data(16) + 2*data(8)*data(11)*data(16) + data(1)*data(16)^2 - 2*data(8)*data(10)*data(17) + 2*data(7)*data(11)*data(17) + 2*data(2)*data(16)*data(17) - data(1)*data(17)^2;
coeffs(17) = 2*data(3)*data(12)*data(19) - 2*data(6)*data(15)*data(19) + 2*data(7)*data(16)*data(19) - 2*data(8)*data(17)*data(19) + 2*data(8)*data(16)*data(20) + 2*data(7)*data(17)*data(20) + 2*data(3)*data(10)*data(21) + 2*data(1)*data(12)*data(21) + 2*data(6)*data(13)*data(21) + 2*data(4)*data(15)*data(21) + 2*data(6)*data(12)*data(22) + 2*data(3)*data(15)*data(22) - 2*data(6)*data(10)*data(24) + 2*data(4)*data(12)*data(24) + 2*data(3)*data(13)*data(24) - 2*data(1)*data(15)*data(24) + 2*data(7)*data(10)*data(25) + 2*data(8)*data(11)*data(25) + 2*data(1)*data(16)*data(25) + 2*data(2)*data(17)*data(25) - 2*data(8)*data(10)*data(26) + 2*data(7)*data(11)*data(26) + 2*data(2)*data(16)*data(26) - 2*data(1)*data(17)*data(26);
coeffs(18) = 2*data(3)*data(19)*data(21) + data(1)*data(21)^2 + 2*data(6)*data(21)*data(22) - 2*data(6)*data(19)*data(24) + 2*data(4)*data(21)*data(24) + 2*data(3)*data(22)*data(24) - data(1)*data(24)^2 + 2*data(7)*data(19)*data(25) + 2*data(8)*data(20)*data(25) + data(1)*data(25)^2 - 2*data(8)*data(19)*data(26) + 2*data(7)*data(20)*data(26) + 2*data(2)*data(25)*data(26) - data(1)*data(26)^2;
coeffs(19) = -data(9)^2*data(10) + 2*data(7)*data(9)*data(12) + 2*data(3)*data(9)*data(16) + 2*data(3)*data(7)*data(18) - 2*data(1)*data(9)*data(18);
coeffs(20) = -data(9)^2*data(19) + 2*data(7)*data(9)*data(21) + 2*data(3)*data(9)*data(25) + 2*data(3)*data(7)*data(27) - 2*data(1)*data(9)*data(27);
coeffs(21) = 3*data(1)*data(10)^2 + 2*data(2)*data(10)*data(11) + data(1)*data(11)^2 + 2*data(4)*data(10)*data(13) + 2*data(5)*data(11)*data(13) + data(1)*data(13)^2 - 2*data(5)*data(10)*data(14) + 2*data(4)*data(11)*data(14) + 2*data(2)*data(13)*data(14) - data(1)*data(14)^2;
coeffs(22) = 6*data(1)*data(10)*data(19) + 2*data(2)*data(11)*data(19) + 2*data(4)*data(13)*data(19) - 2*data(5)*data(14)*data(19) + 2*data(2)*data(10)*data(20) + 2*data(1)*data(11)*data(20) + 2*data(5)*data(13)*data(20) + 2*data(4)*data(14)*data(20) + 2*data(4)*data(10)*data(22) + 2*data(5)*data(11)*data(22) + 2*data(1)*data(13)*data(22) + 2*data(2)*data(14)*data(22) - 2*data(5)*data(10)*data(23) + 2*data(4)*data(11)*data(23) + 2*data(2)*data(13)*data(23) - 2*data(1)*data(14)*data(23);
coeffs(23) = 3*data(1)*data(19)^2 + 2*data(2)*data(19)*data(20) + data(1)*data(20)^2 + 2*data(4)*data(19)*data(22) + 2*data(5)*data(20)*data(22) + data(1)*data(22)^2 - 2*data(5)*data(19)*data(23) + 2*data(4)*data(20)*data(23) + 2*data(2)*data(22)*data(23) - data(1)*data(23)^2;
coeffs(24) = data(3)^2*data(10) - data(6)^2*data(10) + data(7)^2*data(10) - data(8)^2*data(10) + 2*data(7)*data(8)*data(11) + 2*data(1)*data(3)*data(12) + 2*data(4)*data(6)*data(12) + 2*data(3)*data(6)*data(13) + 2*data(3)*data(4)*data(15) - 2*data(1)*data(6)*data(15) + 2*data(1)*data(7)*data(16) + 2*data(2)*data(8)*data(16) + 2*data(2)*data(7)*data(17) - 2*data(1)*data(8)*data(17);
coeffs(25) = data(3)^2*data(19) - data(6)^2*data(19) + data(7)^2*data(19) - data(8)^2*data(19) + 2*data(7)*data(8)*data(20) + 2*data(1)*data(3)*data(21) + 2*data(4)*data(6)*data(21) + 2*data(3)*data(6)*data(22) + 2*data(3)*data(4)*data(24) - 2*data(1)*data(6)*data(24) + 2*data(1)*data(7)*data(25) + 2*data(2)*data(8)*data(25) + 2*data(2)*data(7)*data(26) - 2*data(1)*data(8)*data(26);
coeffs(26) = 2*data(3)*data(7)*data(9) - data(1)*data(9)^2;
coeffs(27) = 3*data(1)^2*data(10) + data(2)^2*data(10) + data(4)^2*data(10) - data(5)^2*data(10) + 2*data(1)*data(2)*data(11) + 2*data(4)*data(5)*data(11) + 2*data(1)*data(4)*data(13) + 2*data(2)*data(5)*data(13) + 2*data(2)*data(4)*data(14) - 2*data(1)*data(5)*data(14);
coeffs(28) = 3*data(1)^2*data(19) + data(2)^2*data(19) + data(4)^2*data(19) - data(5)^2*data(19) + 2*data(1)*data(2)*data(20) + 2*data(4)*data(5)*data(20) + 2*data(1)*data(4)*data(22) + 2*data(2)*data(5)*data(22) + 2*data(2)*data(4)*data(23) - 2*data(1)*data(5)*data(23);
coeffs(29) = data(1)*data(3)^2 + 2*data(3)*data(4)*data(6) - data(1)*data(6)^2 + data(1)*data(7)^2 + 2*data(2)*data(7)*data(8) - data(1)*data(8)^2;
coeffs(30) = data(1)^3 + data(1)*data(2)^2 + data(1)*data(4)^2 + 2*data(2)*data(4)*data(5) - data(1)*data(5)^2;
coeffs(31) = 2*data(12)*data(17)*data(18) - data(11)*data(18)^2;
coeffs(32) = -data(18)^2*data(20) + 2*data(17)*data(18)*data(21) + 2*data(12)*data(18)*data(26) + 2*data(12)*data(17)*data(27) - 2*data(11)*data(18)*data(27);
coeffs(33) = 2*data(18)*data(21)*data(26) - 2*data(18)*data(20)*data(27) + 2*data(17)*data(21)*data(27) + 2*data(12)*data(26)*data(27) - data(11)*data(27)^2;
coeffs(34) = 2*data(21)*data(26)*data(27) - data(20)*data(27)^2;
coeffs(35) = data(11)*data(12)^2 + 2*data(12)*data(14)*data(15) - data(11)*data(15)^2 - data(11)*data(16)^2 + 2*data(10)*data(16)*data(17) + data(11)*data(17)^2;
coeffs(36) = 2*data(16)*data(17)*data(19) + data(12)^2*data(20) - data(15)^2*data(20) - data(16)^2*data(20) + data(17)^2*data(20) + 2*data(11)*data(12)*data(21) + 2*data(14)*data(15)*data(21) + 2*data(12)*data(15)*data(23) + 2*data(12)*data(14)*data(24) - 2*data(11)*data(15)*data(24) - 2*data(11)*data(16)*data(25) + 2*data(10)*data(17)*data(25) + 2*data(10)*data(16)*data(26) + 2*data(11)*data(17)*data(26);
coeffs(37) = 2*data(12)*data(20)*data(21) + data(11)*data(21)^2 + 2*data(15)*data(21)*data(23) - 2*data(15)*data(20)*data(24) + 2*data(14)*data(21)*data(24) + 2*data(12)*data(23)*data(24) - data(11)*data(24)^2 + 2*data(17)*data(19)*data(25) - 2*data(16)*data(20)*data(25) - data(11)*data(25)^2 + 2*data(16)*data(19)*data(26) + 2*data(17)*data(20)*data(26) + 2*data(10)*data(25)*data(26) + data(11)*data(26)^2;
coeffs(38) = data(20)*data(21)^2 + 2*data(21)*data(23)*data(24) - data(20)*data(24)^2 - data(20)*data(25)^2 + 2*data(19)*data(25)*data(26) + data(20)*data(26)^2;
coeffs(39) = 2*data(9)*data(12)*data(17) - 2*data(9)*data(11)*data(18) + 2*data(8)*data(12)*data(18) + 2*data(3)*data(17)*data(18) - data(2)*data(18)^2;
coeffs(40) = -2*data(9)*data(18)*data(20) + 2*data(9)*data(17)*data(21) + 2*data(8)*data(18)*data(21) + 2*data(9)*data(12)*data(26) + 2*data(3)*data(18)*data(26) - 2*data(9)*data(11)*data(27) + 2*data(8)*data(12)*data(27) + 2*data(3)*data(17)*data(27) - 2*data(2)*data(18)*data(27);
coeffs(41) = 2*data(9)*data(21)*data(26) - 2*data(9)*data(20)*data(27) + 2*data(8)*data(21)*data(27) + 2*data(3)*data(26)*data(27) - data(2)*data(27)^2;
coeffs(42) = data(10)^2*data(11) + data(11)^3 - data(11)*data(13)^2 + 2*data(10)*data(13)*data(14) + data(11)*data(14)^2;
coeffs(43) = 2*data(10)*data(11)*data(19) + 2*data(13)*data(14)*data(19) + data(10)^2*data(20) + 3*data(11)^2*data(20) - data(13)^2*data(20) + data(14)^2*data(20) - 2*data(11)*data(13)*data(22) + 2*data(10)*data(14)*data(22) + 2*data(10)*data(13)*data(23) + 2*data(11)*data(14)*data(23);
coeffs(44) = data(11)*data(19)^2 + 2*data(10)*data(19)*data(20) + 3*data(11)*data(20)^2 + 2*data(14)*data(19)*data(22) - 2*data(13)*data(20)*data(22) - data(11)*data(22)^2 + 2*data(13)*data(19)*data(23) + 2*data(14)*data(20)*data(23) + 2*data(10)*data(22)*data(23) + data(11)*data(23)^2;
coeffs(45) = data(19)^2*data(20) + data(20)^3 - data(20)*data(22)^2 + 2*data(19)*data(22)*data(23) + data(20)*data(23)^2;
coeffs(46) = 2*data(3)*data(11)*data(12) + data(2)*data(12)^2 + 2*data(6)*data(12)*data(14) - 2*data(6)*data(11)*data(15) + 2*data(5)*data(12)*data(15) + 2*data(3)*data(14)*data(15) - data(2)*data(15)^2 + 2*data(8)*data(10)*data(16) - 2*data(7)*data(11)*data(16) - data(2)*data(16)^2 + 2*data(7)*data(10)*data(17) + 2*data(8)*data(11)*data(17) + 2*data(1)*data(16)*data(17) + data(2)*data(17)^2;
coeffs(47) = 2*data(8)*data(16)*data(19) + 2*data(7)*data(17)*data(19) + 2*data(3)*data(12)*data(20) - 2*data(6)*data(15)*data(20) - 2*data(7)*data(16)*data(20) + 2*data(8)*data(17)*data(20) + 2*data(3)*data(11)*data(21) + 2*data(2)*data(12)*data(21) + 2*data(6)*data(14)*data(21) + 2*data(5)*data(15)*data(21) + 2*data(6)*data(12)*data(23) + 2*data(3)*data(15)*data(23) - 2*data(6)*data(11)*data(24) + 2*data(5)*data(12)*data(24) + 2*data(3)*data(14)*data(24) - 2*data(2)*data(15)*data(24) + 2*data(8)*data(10)*data(25) - 2*data(7)*data(11)*data(25) - 2*data(2)*data(16)*data(25) + 2*data(1)*data(17)*data(25) + 2*data(7)*data(10)*data(26) + 2*data(8)*data(11)*data(26) + 2*data(1)*data(16)*data(26) + 2*data(2)*data(17)*data(26);
coeffs(48) = 2*data(3)*data(20)*data(21) + data(2)*data(21)^2 + 2*data(6)*data(21)*data(23) - 2*data(6)*data(20)*data(24) + 2*data(5)*data(21)*data(24) + 2*data(3)*data(23)*data(24) - data(2)*data(24)^2 + 2*data(8)*data(19)*data(25) - 2*data(7)*data(20)*data(25) - data(2)*data(25)^2 + 2*data(7)*data(19)*data(26) + 2*data(8)*data(20)*data(26) + 2*data(1)*data(25)*data(26) + data(2)*data(26)^2;
coeffs(49) = -data(9)^2*data(11) + 2*data(8)*data(9)*data(12) + 2*data(3)*data(9)*data(17) + 2*data(3)*data(8)*data(18) - 2*data(2)*data(9)*data(18);
coeffs(50) = -data(9)^2*data(20) + 2*data(8)*data(9)*data(21) + 2*data(3)*data(9)*data(26) + 2*data(3)*data(8)*data(27) - 2*data(2)*data(9)*data(27);
coeffs(51) = data(2)*data(10)^2 + 2*data(1)*data(10)*data(11) + 3*data(2)*data(11)^2 + 2*data(5)*data(10)*data(13) - 2*data(4)*data(11)*data(13) - data(2)*data(13)^2 + 2*data(4)*data(10)*data(14) + 2*data(5)*data(11)*data(14) + 2*data(1)*data(13)*data(14) + data(2)*data(14)^2;
coeffs(52) = 2*data(2)*data(10)*data(19) + 2*data(1)*data(11)*data(19) + 2*data(5)*data(13)*data(19) + 2*data(4)*data(14)*data(19) + 2*data(1)*data(10)*data(20) + 6*data(2)*data(11)*data(20) - 2*data(4)*data(13)*data(20) + 2*data(5)*data(14)*data(20) + 2*data(5)*data(10)*data(22) - 2*data(4)*data(11)*data(22) - 2*data(2)*data(13)*data(22) + 2*data(1)*data(14)*data(22) + 2*data(4)*data(10)*data(23) + 2*data(5)*data(11)*data(23) + 2*data(1)*data(13)*data(23) + 2*data(2)*data(14)*data(23);
coeffs(53) = data(2)*data(19)^2 + 2*data(1)*data(19)*data(20) + 3*data(2)*data(20)^2 + 2*data(5)*data(19)*data(22) - 2*data(4)*data(20)*data(22) - data(2)*data(22)^2 + 2*data(4)*data(19)*data(23) + 2*data(5)*data(20)*data(23) + 2*data(1)*data(22)*data(23) + data(2)*data(23)^2;
coeffs(54) = 2*data(7)*data(8)*data(10) + data(3)^2*data(11) - data(6)^2*data(11) - data(7)^2*data(11) + data(8)^2*data(11) + 2*data(2)*data(3)*data(12) + 2*data(5)*data(6)*data(12) + 2*data(3)*data(6)*data(14) + 2*data(3)*data(5)*data(15) - 2*data(2)*data(6)*data(15) - 2*data(2)*data(7)*data(16) + 2*data(1)*data(8)*data(16) + 2*data(1)*data(7)*data(17) + 2*data(2)*data(8)*data(17);
coeffs(55) = 2*data(7)*data(8)*data(19) + data(3)^2*data(20) - data(6)^2*data(20) - data(7)^2*data(20) + data(8)^2*data(20) + 2*data(2)*data(3)*data(21) + 2*data(5)*data(6)*data(21) + 2*data(3)*data(6)*data(23) + 2*data(3)*data(5)*data(24) - 2*data(2)*data(6)*data(24) - 2*data(2)*data(7)*data(25) + 2*data(1)*data(8)*data(25) + 2*data(1)*data(7)*data(26) + 2*data(2)*data(8)*data(26);
coeffs(56) = 2*data(3)*data(8)*data(9) - data(2)*data(9)^2;
coeffs(57) = 2*data(1)*data(2)*data(10) + 2*data(4)*data(5)*data(10) + data(1)^2*data(11) + 3*data(2)^2*data(11) - data(4)^2*data(11) + data(5)^2*data(11) - 2*data(2)*data(4)*data(13) + 2*data(1)*data(5)*data(13) + 2*data(1)*data(4)*data(14) + 2*data(2)*data(5)*data(14);
coeffs(58) = 2*data(1)*data(2)*data(19) + 2*data(4)*data(5)*data(19) + data(1)^2*data(20) + 3*data(2)^2*data(20) - data(4)^2*data(20) + data(5)^2*data(20) - 2*data(2)*data(4)*data(22) + 2*data(1)*data(5)*data(22) + 2*data(1)*data(4)*data(23) + 2*data(2)*data(5)*data(23);
coeffs(59) = data(2)*data(3)^2 + 2*data(3)*data(5)*data(6) - data(2)*data(6)^2 - data(2)*data(7)^2 + 2*data(1)*data(7)*data(8) + data(2)*data(8)^2;
coeffs(60) = data(1)^2*data(2) + data(2)^3 - data(2)*data(4)^2 + 2*data(1)*data(4)*data(5) + data(2)*data(5)^2;
coeffs(61) = data(12)*data(18)^2;
coeffs(62) = data(18)^2*data(21) + 2*data(12)*data(18)*data(27);
coeffs(63) = 2*data(18)*data(21)*data(27) + data(12)*data(27)^2;
coeffs(64) = data(21)*data(27)^2;
coeffs(65) = data(12)^3 + data(12)*data(15)^2 - data(12)*data(16)^2 - data(12)*data(17)^2 + 2*data(10)*data(16)*data(18) + 2*data(11)*data(17)*data(18);
coeffs(66) = 2*data(16)*data(18)*data(19) + 2*data(17)*data(18)*data(20) + 3*data(12)^2*data(21) + data(15)^2*data(21) - data(16)^2*data(21) - data(17)^2*data(21) + 2*data(12)*data(15)*data(24) - 2*data(12)*data(16)*data(25) + 2*data(10)*data(18)*data(25) - 2*data(12)*data(17)*data(26) + 2*data(11)*data(18)*data(26) + 2*data(10)*data(16)*data(27) + 2*data(11)*data(17)*data(27);
coeffs(67) = 3*data(12)*data(21)^2 + 2*data(15)*data(21)*data(24) + data(12)*data(24)^2 + 2*data(18)*data(19)*data(25) - 2*data(16)*data(21)*data(25) - data(12)*data(25)^2 + 2*data(18)*data(20)*data(26) - 2*data(17)*data(21)*data(26) - data(12)*data(26)^2 + 2*data(16)*data(19)*data(27) + 2*data(17)*data(20)*data(27) + 2*data(10)*data(25)*data(27) + 2*data(11)*data(26)*data(27);
coeffs(68) = data(21)^3 + data(21)*data(24)^2 - data(21)*data(25)^2 - data(21)*data(26)^2 + 2*data(19)*data(25)*data(27) + 2*data(20)*data(26)*data(27);
coeffs(69) = 2*data(9)*data(12)*data(18) + data(3)*data(18)^2;
coeffs(70) = 2*data(9)*data(18)*data(21) + 2*data(9)*data(12)*data(27) + 2*data(3)*data(18)*data(27);
coeffs(71) = 2*data(9)*data(21)*data(27) + data(3)*data(27)^2;
coeffs(72) = data(10)^2*data(12) + data(11)^2*data(12) - data(12)*data(13)^2 - data(12)*data(14)^2 + 2*data(10)*data(13)*data(15) + 2*data(11)*data(14)*data(15);
coeffs(73) = 2*data(10)*data(12)*data(19) + 2*data(13)*data(15)*data(19) + 2*data(11)*data(12)*data(20) + 2*data(14)*data(15)*data(20) + data(10)^2*data(21) + data(11)^2*data(21) - data(13)^2*data(21) - data(14)^2*data(21) - 2*data(12)*data(13)*data(22) + 2*data(10)*data(15)*data(22) - 2*data(12)*data(14)*data(23) + 2*data(11)*data(15)*data(23) + 2*data(10)*data(13)*data(24) + 2*data(11)*data(14)*data(24);
coeffs(74) = data(12)*data(19)^2 + data(12)*data(20)^2 + 2*data(10)*data(19)*data(21) + 2*data(11)*data(20)*data(21) + 2*data(15)*data(19)*data(22) - 2*data(13)*data(21)*data(22) - data(12)*data(22)^2 + 2*data(15)*data(20)*data(23) - 2*data(14)*data(21)*data(23) - data(12)*data(23)^2 + 2*data(13)*data(19)*data(24) + 2*data(14)*data(20)*data(24) + 2*data(10)*data(22)*data(24) + 2*data(11)*data(23)*data(24);
coeffs(75) = data(19)^2*data(21) + data(20)^2*data(21) - data(21)*data(22)^2 - data(21)*data(23)^2 + 2*data(19)*data(22)*data(24) + 2*data(20)*data(23)*data(24);
coeffs(76) = 3*data(3)*data(12)^2 + 2*data(6)*data(12)*data(15) + data(3)*data(15)^2 + 2*data(9)*data(10)*data(16) - 2*data(7)*data(12)*data(16) - data(3)*data(16)^2 + 2*data(9)*data(11)*data(17) - 2*data(8)*data(12)*data(17) - data(3)*data(17)^2 + 2*data(7)*data(10)*data(18) + 2*data(8)*data(11)*data(18) + 2*data(1)*data(16)*data(18) + 2*data(2)*data(17)*data(18);
coeffs(77) = 2*data(9)*data(16)*data(19) + 2*data(7)*data(18)*data(19) + 2*data(9)*data(17)*data(20) + 2*data(8)*data(18)*data(20) + 6*data(3)*data(12)*data(21) + 2*data(6)*data(15)*data(21) - 2*data(7)*data(16)*data(21) - 2*data(8)*data(17)*data(21) + 2*data(6)*data(12)*data(24) + 2*data(3)*data(15)*data(24) + 2*data(9)*data(10)*data(25) - 2*data(7)*data(12)*data(25) - 2*data(3)*data(16)*data(25) + 2*data(1)*data(18)*data(25) + 2*data(9)*data(11)*data(26) - 2*data(8)*data(12)*data(26) - 2*data(3)*data(17)*data(26) + 2*data(2)*data(18)*data(26) + 2*data(7)*data(10)*data(27) + 2*data(8)*data(11)*data(27) + 2*data(1)*data(16)*data(27) + 2*data(2)*data(17)*data(27);
coeffs(78) = 3*data(3)*data(21)^2 + 2*data(6)*data(21)*data(24) + data(3)*data(24)^2 + 2*data(9)*data(19)*data(25) - 2*data(7)*data(21)*data(25) - data(3)*data(25)^2 + 2*data(9)*data(20)*data(26) - 2*data(8)*data(21)*data(26) - data(3)*data(26)^2 + 2*data(7)*data(19)*data(27) + 2*data(8)*data(20)*data(27) + 2*data(1)*data(25)*data(27) + 2*data(2)*data(26)*data(27);
coeffs(79) = data(9)^2*data(12) + 2*data(3)*data(9)*data(18);
coeffs(80) = data(9)^2*data(21) + 2*data(3)*data(9)*data(27);
coeffs(81) = data(3)*data(10)^2 + data(3)*data(11)^2 + 2*data(1)*data(10)*data(12) + 2*data(2)*data(11)*data(12) + 2*data(6)*data(10)*data(13) - 2*data(4)*data(12)*data(13) - data(3)*data(13)^2 + 2*data(6)*data(11)*data(14) - 2*data(5)*data(12)*data(14) - data(3)*data(14)^2 + 2*data(4)*data(10)*data(15) + 2*data(5)*data(11)*data(15) + 2*data(1)*data(13)*data(15) + 2*data(2)*data(14)*data(15);
coeffs(82) = 2*data(3)*data(10)*data(19) + 2*data(1)*data(12)*data(19) + 2*data(6)*data(13)*data(19) + 2*data(4)*data(15)*data(19) + 2*data(3)*data(11)*data(20) + 2*data(2)*data(12)*data(20) + 2*data(6)*data(14)*data(20) + 2*data(5)*data(15)*data(20) + 2*data(1)*data(10)*data(21) + 2*data(2)*data(11)*data(21) - 2*data(4)*data(13)*data(21) - 2*data(5)*data(14)*data(21) + 2*data(6)*data(10)*data(22) - 2*data(4)*data(12)*data(22) - 2*data(3)*data(13)*data(22) + 2*data(1)*data(15)*data(22) + 2*data(6)*data(11)*data(23) - 2*data(5)*data(12)*data(23) - 2*data(3)*data(14)*data(23) + 2*data(2)*data(15)*data(23) + 2*data(4)*data(10)*data(24) + 2*data(5)*data(11)*data(24) + 2*data(1)*data(13)*data(24) + 2*data(2)*data(14)*data(24);
coeffs(83) = data(3)*data(19)^2 + data(3)*data(20)^2 + 2*data(1)*data(19)*data(21) + 2*data(2)*data(20)*data(21) + 2*data(6)*data(19)*data(22) - 2*data(4)*data(21)*data(22) - data(3)*data(22)^2 + 2*data(6)*data(20)*data(23) - 2*data(5)*data(21)*data(23) - data(3)*data(23)^2 + 2*data(4)*data(19)*data(24) + 2*data(5)*data(20)*data(24) + 2*data(1)*data(22)*data(24) + 2*data(2)*data(23)*data(24);
coeffs(84) = 2*data(7)*data(9)*data(10) + 2*data(8)*data(9)*data(11) + 3*data(3)^2*data(12) + data(6)^2*data(12) - data(7)^2*data(12) - data(8)^2*data(12) + 2*data(3)*data(6)*data(15) - 2*data(3)*data(7)*data(16) + 2*data(1)*data(9)*data(16) - 2*data(3)*data(8)*data(17) + 2*data(2)*data(9)*data(17) + 2*data(1)*data(7)*data(18) + 2*data(2)*data(8)*data(18);
coeffs(85) = 2*data(7)*data(9)*data(19) + 2*data(8)*data(9)*data(20) + 3*data(3)^2*data(21) + data(6)^2*data(21) - data(7)^2*data(21) - data(8)^2*data(21) + 2*data(3)*data(6)*data(24) - 2*data(3)*data(7)*data(25) + 2*data(1)*data(9)*data(25) - 2*data(3)*data(8)*data(26) + 2*data(2)*data(9)*data(26) + 2*data(1)*data(7)*data(27) + 2*data(2)*data(8)*data(27);
coeffs(86) = data(3)*data(9)^2;
coeffs(87) = 2*data(1)*data(3)*data(10) + 2*data(4)*data(6)*data(10) + 2*data(2)*data(3)*data(11) + 2*data(5)*data(6)*data(11) + data(1)^2*data(12) + data(2)^2*data(12) - data(4)^2*data(12) - data(5)^2*data(12) - 2*data(3)*data(4)*data(13) + 2*data(1)*data(6)*data(13) - 2*data(3)*data(5)*data(14) + 2*data(2)*data(6)*data(14) + 2*data(1)*data(4)*data(15) + 2*data(2)*data(5)*data(15);
coeffs(88) = 2*data(1)*data(3)*data(19) + 2*data(4)*data(6)*data(19) + 2*data(2)*data(3)*data(20) + 2*data(5)*data(6)*data(20) + data(1)^2*data(21) + data(2)^2*data(21) - data(4)^2*data(21) - data(5)^2*data(21) - 2*data(3)*data(4)*data(22) + 2*data(1)*data(6)*data(22) - 2*data(3)*data(5)*data(23) + 2*data(2)*data(6)*data(23) + 2*data(1)*data(4)*data(24) + 2*data(2)*data(5)*data(24);
coeffs(89) = data(3)^3 + data(3)*data(6)^2 - data(3)*data(7)^2 - data(3)*data(8)^2 + 2*data(1)*data(7)*data(9) + 2*data(2)*data(8)*data(9);
coeffs(90) = data(1)^2*data(3) + data(2)^2*data(3) - data(3)*data(4)^2 - data(3)*data(5)^2 + 2*data(1)*data(4)*data(6) + 2*data(2)*data(5)*data(6);
coeffs(91) = 2*data(15)*data(16)*data(18) - data(13)*data(18)^2;
coeffs(92) = -data(18)^2*data(22) + 2*data(16)*data(18)*data(24) + 2*data(15)*data(18)*data(25) + 2*data(15)*data(16)*data(27) - 2*data(13)*data(18)*data(27);
coeffs(93) = 2*data(18)*data(24)*data(25) - 2*data(18)*data(22)*data(27) + 2*data(16)*data(24)*data(27) + 2*data(15)*data(25)*data(27) - data(13)*data(27)^2;
coeffs(94) = 2*data(24)*data(25)*data(27) - data(22)*data(27)^2;
coeffs(95) = -data(12)^2*data(13) + 2*data(10)*data(12)*data(15) + data(13)*data(15)^2 + data(13)*data(16)^2 + 2*data(14)*data(16)*data(17) - data(13)*data(17)^2;
coeffs(96) = 2*data(12)*data(15)*data(19) - 2*data(12)*data(13)*data(21) + 2*data(10)*data(15)*data(21) - data(12)^2*data(22) + data(15)^2*data(22) + data(16)^2*data(22) - data(17)^2*data(22) + 2*data(16)*data(17)*data(23) + 2*data(10)*data(12)*data(24) + 2*data(13)*data(15)*data(24) + 2*data(13)*data(16)*data(25) + 2*data(14)*data(17)*data(25) + 2*data(14)*data(16)*data(26) - 2*data(13)*data(17)*data(26);
coeffs(97) = 2*data(15)*data(19)*data(21) - data(13)*data(21)^2 - 2*data(12)*data(21)*data(22) + 2*data(12)*data(19)*data(24) + 2*data(10)*data(21)*data(24) + 2*data(15)*data(22)*data(24) + data(13)*data(24)^2 + 2*data(16)*data(22)*data(25) + 2*data(17)*data(23)*data(25) + data(13)*data(25)^2 - 2*data(17)*data(22)*data(26) + 2*data(16)*data(23)*data(26) + 2*data(14)*data(25)*data(26) - data(13)*data(26)^2;
coeffs(98) = -data(21)^2*data(22) + 2*data(19)*data(21)*data(24) + data(22)*data(24)^2 + data(22)*data(25)^2 + 2*data(23)*data(25)*data(26) - data(22)*data(26)^2;
coeffs(99) = 2*data(9)*data(15)*data(16) - 2*data(9)*data(13)*data(18) + 2*data(7)*data(15)*data(18) + 2*data(6)*data(16)*data(18) - data(4)*data(18)^2;
coeffs(100) = -2*data(9)*data(18)*data(22) + 2*data(9)*data(16)*data(24) + 2*data(7)*data(18)*data(24) + 2*data(9)*data(15)*data(25) + 2*data(6)*data(18)*data(25) - 2*data(9)*data(13)*data(27) + 2*data(7)*data(15)*data(27) + 2*data(6)*data(16)*data(27) - 2*data(4)*data(18)*data(27);
coeffs(101) = 2*data(9)*data(24)*data(25) - 2*data(9)*data(22)*data(27) + 2*data(7)*data(24)*data(27) + 2*data(6)*data(25)*data(27) - data(4)*data(27)^2;
coeffs(102) = data(10)^2*data(13) - data(11)^2*data(13) + data(13)^3 + 2*data(10)*data(11)*data(14) + data(13)*data(14)^2;
coeffs(103) = 2*data(10)*data(13)*data(19) + 2*data(11)*data(14)*data(19) - 2*data(11)*data(13)*data(20) + 2*data(10)*data(14)*data(20) + data(10)^2*data(22) - data(11)^2*data(22) + 3*data(13)^2*data(22) + data(14)^2*data(22) + 2*data(10)*data(11)*data(23) + 2*data(13)*data(14)*data(23);
coeffs(104) = data(13)*data(19)^2 + 2*data(14)*data(19)*data(20) - data(13)*data(20)^2 + 2*data(10)*data(19)*data(22) - 2*data(11)*data(20)*data(22) + 3*data(13)*data(22)^2 + 2*data(11)*data(19)*data(23) + 2*data(10)*data(20)*data(23) + 2*data(14)*data(22)*data(23) + data(13)*data(23)^2;
coeffs(105) = data(19)^2*data(22) - data(20)^2*data(22) + data(22)^3 + 2*data(19)*data(20)*data(23) + data(22)*data(23)^2;
coeffs(106) = 2*data(6)*data(10)*data(12) - data(4)*data(12)^2 - 2*data(3)*data(12)*data(13) + 2*data(3)*data(10)*data(15) + 2*data(1)*data(12)*data(15) + 2*data(6)*data(13)*data(15) + data(4)*data(15)^2 + 2*data(7)*data(13)*data(16) + 2*data(8)*data(14)*data(16) + data(4)*data(16)^2 - 2*data(8)*data(13)*data(17) + 2*data(7)*data(14)*data(17) + 2*data(5)*data(16)*data(17) - data(4)*data(17)^2;
coeffs(107) = 2*data(6)*data(12)*data(19) + 2*data(3)*data(15)*data(19) + 2*data(6)*data(10)*data(21) - 2*data(4)*data(12)*data(21) - 2*data(3)*data(13)*data(21) + 2*data(1)*data(15)*data(21) - 2*data(3)*data(12)*data(22) + 2*data(6)*data(15)*data(22) + 2*data(7)*data(16)*data(22) - 2*data(8)*data(17)*data(22) + 2*data(8)*data(16)*data(23) + 2*data(7)*data(17)*data(23) + 2*data(3)*data(10)*data(24) + 2*data(1)*data(12)*data(24) + 2*data(6)*data(13)*data(24) + 2*data(4)*data(15)*data(24) + 2*data(7)*data(13)*data(25) + 2*data(8)*data(14)*data(25) + 2*data(4)*data(16)*data(25) + 2*data(5)*data(17)*data(25) - 2*data(8)*data(13)*data(26) + 2*data(7)*data(14)*data(26) + 2*data(5)*data(16)*data(26) - 2*data(4)*data(17)*data(26);
coeffs(108) = 2*data(6)*data(19)*data(21) - data(4)*data(21)^2 - 2*data(3)*data(21)*data(22) + 2*data(3)*data(19)*data(24) + 2*data(1)*data(21)*data(24) + 2*data(6)*data(22)*data(24) + data(4)*data(24)^2 + 2*data(7)*data(22)*data(25) + 2*data(8)*data(23)*data(25) + data(4)*data(25)^2 - 2*data(8)*data(22)*data(26) + 2*data(7)*data(23)*data(26) + 2*data(5)*data(25)*data(26) - data(4)*data(26)^2;
coeffs(109) = -data(9)^2*data(13) + 2*data(7)*data(9)*data(15) + 2*data(6)*data(9)*data(16) + 2*data(6)*data(7)*data(18) - 2*data(4)*data(9)*data(18);
coeffs(110) = -data(9)^2*data(22) + 2*data(7)*data(9)*data(24) + 2*data(6)*data(9)*data(25) + 2*data(6)*data(7)*data(27) - 2*data(4)*data(9)*data(27);
coeffs(111) = data(4)*data(10)^2 + 2*data(5)*data(10)*data(11) - data(4)*data(11)^2 + 2*data(1)*data(10)*data(13) - 2*data(2)*data(11)*data(13) + 3*data(4)*data(13)^2 + 2*data(2)*data(10)*data(14) + 2*data(1)*data(11)*data(14) + 2*data(5)*data(13)*data(14) + data(4)*data(14)^2;
coeffs(112) = 2*data(4)*data(10)*data(19) + 2*data(5)*data(11)*data(19) + 2*data(1)*data(13)*data(19) + 2*data(2)*data(14)*data(19) + 2*data(5)*data(10)*data(20) - 2*data(4)*data(11)*data(20) - 2*data(2)*data(13)*data(20) + 2*data(1)*data(14)*data(20) + 2*data(1)*data(10)*data(22) - 2*data(2)*data(11)*data(22) + 6*data(4)*data(13)*data(22) + 2*data(5)*data(14)*data(22) + 2*data(2)*data(10)*data(23) + 2*data(1)*data(11)*data(23) + 2*data(5)*data(13)*data(23) + 2*data(4)*data(14)*data(23);
coeffs(113) = data(4)*data(19)^2 + 2*data(5)*data(19)*data(20) - data(4)*data(20)^2 + 2*data(1)*data(19)*data(22) - 2*data(2)*data(20)*data(22) + 3*data(4)*data(22)^2 + 2*data(2)*data(19)*data(23) + 2*data(1)*data(20)*data(23) + 2*data(5)*data(22)*data(23) + data(4)*data(23)^2;
coeffs(114) = 2*data(3)*data(6)*data(10) - 2*data(3)*data(4)*data(12) + 2*data(1)*data(6)*data(12) - data(3)^2*data(13) + data(6)^2*data(13) + data(7)^2*data(13) - data(8)^2*data(13) + 2*data(7)*data(8)*data(14) + 2*data(1)*data(3)*data(15) + 2*data(4)*data(6)*data(15) + 2*data(4)*data(7)*data(16) + 2*data(5)*data(8)*data(16) + 2*data(5)*data(7)*data(17) - 2*data(4)*data(8)*data(17);
coeffs(115) = 2*data(3)*data(6)*data(19) - 2*data(3)*data(4)*data(21) + 2*data(1)*data(6)*data(21) - data(3)^2*data(22) + data(6)^2*data(22) + data(7)^2*data(22) - data(8)^2*data(22) + 2*data(7)*data(8)*data(23) + 2*data(1)*data(3)*data(24) + 2*data(4)*data(6)*data(24) + 2*data(4)*data(7)*data(25) + 2*data(5)*data(8)*data(25) + 2*data(5)*data(7)*data(26) - 2*data(4)*data(8)*data(26);
coeffs(116) = 2*data(6)*data(7)*data(9) - data(4)*data(9)^2;
coeffs(117) = 2*data(1)*data(4)*data(10) + 2*data(2)*data(5)*data(10) - 2*data(2)*data(4)*data(11) + 2*data(1)*data(5)*data(11) + data(1)^2*data(13) - data(2)^2*data(13) + 3*data(4)^2*data(13) + data(5)^2*data(13) + 2*data(1)*data(2)*data(14) + 2*data(4)*data(5)*data(14);
coeffs(118) = 2*data(1)*data(4)*data(19) + 2*data(2)*data(5)*data(19) - 2*data(2)*data(4)*data(20) + 2*data(1)*data(5)*data(20) + data(1)^2*data(22) - data(2)^2*data(22) + 3*data(4)^2*data(22) + data(5)^2*data(22) + 2*data(1)*data(2)*data(23) + 2*data(4)*data(5)*data(23);
coeffs(119) = -data(3)^2*data(4) + 2*data(1)*data(3)*data(6) + data(4)*data(6)^2 + data(4)*data(7)^2 + 2*data(5)*data(7)*data(8) - data(4)*data(8)^2;
coeffs(120) = data(1)^2*data(4) - data(2)^2*data(4) + data(4)^3 + 2*data(1)*data(2)*data(5) + data(4)*data(5)^2;
coeffs(121) = 2*data(15)*data(17)*data(18) - data(14)*data(18)^2;
coeffs(122) = -data(18)^2*data(23) + 2*data(17)*data(18)*data(24) + 2*data(15)*data(18)*data(26) + 2*data(15)*data(17)*data(27) - 2*data(14)*data(18)*data(27);
coeffs(123) = 2*data(18)*data(24)*data(26) - 2*data(18)*data(23)*data(27) + 2*data(17)*data(24)*data(27) + 2*data(15)*data(26)*data(27) - data(14)*data(27)^2;
coeffs(124) = 2*data(24)*data(26)*data(27) - data(23)*data(27)^2;
coeffs(125) = -data(12)^2*data(14) + 2*data(11)*data(12)*data(15) + data(14)*data(15)^2 - data(14)*data(16)^2 + 2*data(13)*data(16)*data(17) + data(14)*data(17)^2;
coeffs(126) = 2*data(12)*data(15)*data(20) - 2*data(12)*data(14)*data(21) + 2*data(11)*data(15)*data(21) + 2*data(16)*data(17)*data(22) - data(12)^2*data(23) + data(15)^2*data(23) - data(16)^2*data(23) + data(17)^2*data(23) + 2*data(11)*data(12)*data(24) + 2*data(14)*data(15)*data(24) - 2*data(14)*data(16)*data(25) + 2*data(13)*data(17)*data(25) + 2*data(13)*data(16)*data(26) + 2*data(14)*data(17)*data(26);
coeffs(127) = 2*data(15)*data(20)*data(21) - data(14)*data(21)^2 - 2*data(12)*data(21)*data(23) + 2*data(12)*data(20)*data(24) + 2*data(11)*data(21)*data(24) + 2*data(15)*data(23)*data(24) + data(14)*data(24)^2 + 2*data(17)*data(22)*data(25) - 2*data(16)*data(23)*data(25) - data(14)*data(25)^2 + 2*data(16)*data(22)*data(26) + 2*data(17)*data(23)*data(26) + 2*data(13)*data(25)*data(26) + data(14)*data(26)^2;
coeffs(128) = -data(21)^2*data(23) + 2*data(20)*data(21)*data(24) + data(23)*data(24)^2 - data(23)*data(25)^2 + 2*data(22)*data(25)*data(26) + data(23)*data(26)^2;
coeffs(129) = 2*data(9)*data(15)*data(17) - 2*data(9)*data(14)*data(18) + 2*data(8)*data(15)*data(18) + 2*data(6)*data(17)*data(18) - data(5)*data(18)^2;
coeffs(130) = -2*data(9)*data(18)*data(23) + 2*data(9)*data(17)*data(24) + 2*data(8)*data(18)*data(24) + 2*data(9)*data(15)*data(26) + 2*data(6)*data(18)*data(26) - 2*data(9)*data(14)*data(27) + 2*data(8)*data(15)*data(27) + 2*data(6)*data(17)*data(27) - 2*data(5)*data(18)*data(27);
coeffs(131) = 2*data(9)*data(24)*data(26) - 2*data(9)*data(23)*data(27) + 2*data(8)*data(24)*data(27) + 2*data(6)*data(26)*data(27) - data(5)*data(27)^2;
coeffs(132) = 2*data(10)*data(11)*data(13) - data(10)^2*data(14) + data(11)^2*data(14) + data(13)^2*data(14) + data(14)^3;
coeffs(133) = 2*data(11)*data(13)*data(19) - 2*data(10)*data(14)*data(19) + 2*data(10)*data(13)*data(20) + 2*data(11)*data(14)*data(20) + 2*data(10)*data(11)*data(22) + 2*data(13)*data(14)*data(22) - data(10)^2*data(23) + data(11)^2*data(23) + data(13)^2*data(23) + 3*data(14)^2*data(23);
coeffs(134) = -data(14)*data(19)^2 + 2*data(13)*data(19)*data(20) + data(14)*data(20)^2 + 2*data(11)*data(19)*data(22) + 2*data(10)*data(20)*data(22) + data(14)*data(22)^2 - 2*data(10)*data(19)*data(23) + 2*data(11)*data(20)*data(23) + 2*data(13)*data(22)*data(23) + 3*data(14)*data(23)^2;
coeffs(135) = 2*data(19)*data(20)*data(22) - data(19)^2*data(23) + data(20)^2*data(23) + data(22)^2*data(23) + data(23)^3;
coeffs(136) = 2*data(6)*data(11)*data(12) - data(5)*data(12)^2 - 2*data(3)*data(12)*data(14) + 2*data(3)*data(11)*data(15) + 2*data(2)*data(12)*data(15) + 2*data(6)*data(14)*data(15) + data(5)*data(15)^2 + 2*data(8)*data(13)*data(16) - 2*data(7)*data(14)*data(16) - data(5)*data(16)^2 + 2*data(7)*data(13)*data(17) + 2*data(8)*data(14)*data(17) + 2*data(4)*data(16)*data(17) + data(5)*data(17)^2;
coeffs(137) = 2*data(6)*data(12)*data(20) + 2*data(3)*data(15)*data(20) + 2*data(6)*data(11)*data(21) - 2*data(5)*data(12)*data(21) - 2*data(3)*data(14)*data(21) + 2*data(2)*data(15)*data(21) + 2*data(8)*data(16)*data(22) + 2*data(7)*data(17)*data(22) - 2*data(3)*data(12)*data(23) + 2*data(6)*data(15)*data(23) - 2*data(7)*data(16)*data(23) + 2*data(8)*data(17)*data(23) + 2*data(3)*data(11)*data(24) + 2*data(2)*data(12)*data(24) + 2*data(6)*data(14)*data(24) + 2*data(5)*data(15)*data(24) + 2*data(8)*data(13)*data(25) - 2*data(7)*data(14)*data(25) - 2*data(5)*data(16)*data(25) + 2*data(4)*data(17)*data(25) + 2*data(7)*data(13)*data(26) + 2*data(8)*data(14)*data(26) + 2*data(4)*data(16)*data(26) + 2*data(5)*data(17)*data(26);
coeffs(138) = 2*data(6)*data(20)*data(21) - data(5)*data(21)^2 - 2*data(3)*data(21)*data(23) + 2*data(3)*data(20)*data(24) + 2*data(2)*data(21)*data(24) + 2*data(6)*data(23)*data(24) + data(5)*data(24)^2 + 2*data(8)*data(22)*data(25) - 2*data(7)*data(23)*data(25) - data(5)*data(25)^2 + 2*data(7)*data(22)*data(26) + 2*data(8)*data(23)*data(26) + 2*data(4)*data(25)*data(26) + data(5)*data(26)^2;
coeffs(139) = -data(9)^2*data(14) + 2*data(8)*data(9)*data(15) + 2*data(6)*data(9)*data(17) + 2*data(6)*data(8)*data(18) - 2*data(5)*data(9)*data(18);
coeffs(140) = -data(9)^2*data(23) + 2*data(8)*data(9)*data(24) + 2*data(6)*data(9)*data(26) + 2*data(6)*data(8)*data(27) - 2*data(5)*data(9)*data(27);
coeffs(141) = -data(5)*data(10)^2 + 2*data(4)*data(10)*data(11) + data(5)*data(11)^2 + 2*data(2)*data(10)*data(13) + 2*data(1)*data(11)*data(13) + data(5)*data(13)^2 - 2*data(1)*data(10)*data(14) + 2*data(2)*data(11)*data(14) + 2*data(4)*data(13)*data(14) + 3*data(5)*data(14)^2;
coeffs(142) = -2*data(5)*data(10)*data(19) + 2*data(4)*data(11)*data(19) + 2*data(2)*data(13)*data(19) - 2*data(1)*data(14)*data(19) + 2*data(4)*data(10)*data(20) + 2*data(5)*data(11)*data(20) + 2*data(1)*data(13)*data(20) + 2*data(2)*data(14)*data(20) + 2*data(2)*data(10)*data(22) + 2*data(1)*data(11)*data(22) + 2*data(5)*data(13)*data(22) + 2*data(4)*data(14)*data(22) - 2*data(1)*data(10)*data(23) + 2*data(2)*data(11)*data(23) + 2*data(4)*data(13)*data(23) + 6*data(5)*data(14)*data(23);
coeffs(143) = -data(5)*data(19)^2 + 2*data(4)*data(19)*data(20) + data(5)*data(20)^2 + 2*data(2)*data(19)*data(22) + 2*data(1)*data(20)*data(22) + data(5)*data(22)^2 - 2*data(1)*data(19)*data(23) + 2*data(2)*data(20)*data(23) + 2*data(4)*data(22)*data(23) + 3*data(5)*data(23)^2;
coeffs(144) = 2*data(3)*data(6)*data(11) - 2*data(3)*data(5)*data(12) + 2*data(2)*data(6)*data(12) + 2*data(7)*data(8)*data(13) - data(3)^2*data(14) + data(6)^2*data(14) - data(7)^2*data(14) + data(8)^2*data(14) + 2*data(2)*data(3)*data(15) + 2*data(5)*data(6)*data(15) - 2*data(5)*data(7)*data(16) + 2*data(4)*data(8)*data(16) + 2*data(4)*data(7)*data(17) + 2*data(5)*data(8)*data(17);
coeffs(145) = 2*data(3)*data(6)*data(20) - 2*data(3)*data(5)*data(21) + 2*data(2)*data(6)*data(21) + 2*data(7)*data(8)*data(22) - data(3)^2*data(23) + data(6)^2*data(23) - data(7)^2*data(23) + data(8)^2*data(23) + 2*data(2)*data(3)*data(24) + 2*data(5)*data(6)*data(24) - 2*data(5)*data(7)*data(25) + 2*data(4)*data(8)*data(25) + 2*data(4)*data(7)*data(26) + 2*data(5)*data(8)*data(26);
coeffs(146) = 2*data(6)*data(8)*data(9) - data(5)*data(9)^2;
coeffs(147) = 2*data(2)*data(4)*data(10) - 2*data(1)*data(5)*data(10) + 2*data(1)*data(4)*data(11) + 2*data(2)*data(5)*data(11) + 2*data(1)*data(2)*data(13) + 2*data(4)*data(5)*data(13) - data(1)^2*data(14) + data(2)^2*data(14) + data(4)^2*data(14) + 3*data(5)^2*data(14);
coeffs(148) = 2*data(2)*data(4)*data(19) - 2*data(1)*data(5)*data(19) + 2*data(1)*data(4)*data(20) + 2*data(2)*data(5)*data(20) + 2*data(1)*data(2)*data(22) + 2*data(4)*data(5)*data(22) - data(1)^2*data(23) + data(2)^2*data(23) + data(4)^2*data(23) + 3*data(5)^2*data(23);
coeffs(149) = -data(3)^2*data(5) + 2*data(2)*data(3)*data(6) + data(5)*data(6)^2 - data(5)*data(7)^2 + 2*data(4)*data(7)*data(8) + data(5)*data(8)^2;
coeffs(150) = 2*data(1)*data(2)*data(4) - data(1)^2*data(5) + data(2)^2*data(5) + data(4)^2*data(5) + data(5)^3;
coeffs(151) = data(15)*data(18)^2;
coeffs(152) = data(18)^2*data(24) + 2*data(15)*data(18)*data(27);
coeffs(153) = 2*data(18)*data(24)*data(27) + data(15)*data(27)^2;
coeffs(154) = data(24)*data(27)^2;
coeffs(155) = data(12)^2*data(15) + data(15)^3 - data(15)*data(16)^2 - data(15)*data(17)^2 + 2*data(13)*data(16)*data(18) + 2*data(14)*data(17)*data(18);
coeffs(156) = 2*data(12)*data(15)*data(21) + 2*data(16)*data(18)*data(22) + 2*data(17)*data(18)*data(23) + data(12)^2*data(24) + 3*data(15)^2*data(24) - data(16)^2*data(24) - data(17)^2*data(24) - 2*data(15)*data(16)*data(25) + 2*data(13)*data(18)*data(25) - 2*data(15)*data(17)*data(26) + 2*data(14)*data(18)*data(26) + 2*data(13)*data(16)*data(27) + 2*data(14)*data(17)*data(27);
coeffs(157) = data(15)*data(21)^2 + 2*data(12)*data(21)*data(24) + 3*data(15)*data(24)^2 + 2*data(18)*data(22)*data(25) - 2*data(16)*data(24)*data(25) - data(15)*data(25)^2 + 2*data(18)*data(23)*data(26) - 2*data(17)*data(24)*data(26) - data(15)*data(26)^2 + 2*data(16)*data(22)*data(27) + 2*data(17)*data(23)*data(27) + 2*data(13)*data(25)*data(27) + 2*data(14)*data(26)*data(27);
coeffs(158) = data(21)^2*data(24) + data(24)^3 - data(24)*data(25)^2 - data(24)*data(26)^2 + 2*data(22)*data(25)*data(27) + 2*data(23)*data(26)*data(27);
coeffs(159) = 2*data(9)*data(15)*data(18) + data(6)*data(18)^2;
coeffs(160) = 2*data(9)*data(18)*data(24) + 2*data(9)*data(15)*data(27) + 2*data(6)*data(18)*data(27);
coeffs(161) = 2*data(9)*data(24)*data(27) + data(6)*data(27)^2;
coeffs(162) = 2*data(10)*data(12)*data(13) + 2*data(11)*data(12)*data(14) - data(10)^2*data(15) - data(11)^2*data(15) + data(13)^2*data(15) + data(14)^2*data(15);
coeffs(163) = 2*data(12)*data(13)*data(19) - 2*data(10)*data(15)*data(19) + 2*data(12)*data(14)*data(20) - 2*data(11)*data(15)*data(20) + 2*data(10)*data(13)*data(21) + 2*data(11)*data(14)*data(21) + 2*data(10)*data(12)*data(22) + 2*data(13)*data(15)*data(22) + 2*data(11)*data(12)*data(23) + 2*data(14)*data(15)*data(23) - data(10)^2*data(24) - data(11)^2*data(24) + data(13)^2*data(24) + data(14)^2*data(24);
coeffs(164) = -data(15)*data(19)^2 - data(15)*data(20)^2 + 2*data(13)*data(19)*data(21) + 2*data(14)*data(20)*data(21) + 2*data(12)*data(19)*data(22) + 2*data(10)*data(21)*data(22) + data(15)*data(22)^2 + 2*data(12)*data(20)*data(23) + 2*data(11)*data(21)*data(23) + data(15)*data(23)^2 - 2*data(10)*data(19)*data(24) - 2*data(11)*data(20)*data(24) + 2*data(13)*data(22)*data(24) + 2*data(14)*data(23)*data(24);
coeffs(165) = 2*data(19)*data(21)*data(22) + 2*data(20)*data(21)*data(23) - data(19)^2*data(24) - data(20)^2*data(24) + data(22)^2*data(24) + data(23)^2*data(24);
coeffs(166) = data(6)*data(12)^2 + 2*data(3)*data(12)*data(15) + 3*data(6)*data(15)^2 + 2*data(9)*data(13)*data(16) - 2*data(7)*data(15)*data(16) - data(6)*data(16)^2 + 2*data(9)*data(14)*data(17) - 2*data(8)*data(15)*data(17) - data(6)*data(17)^2 + 2*data(7)*data(13)*data(18) + 2*data(8)*data(14)*data(18) + 2*data(4)*data(16)*data(18) + 2*data(5)*data(17)*data(18);
coeffs(167) = 2*data(6)*data(12)*data(21) + 2*data(3)*data(15)*data(21) + 2*data(9)*data(16)*data(22) + 2*data(7)*data(18)*data(22) + 2*data(9)*data(17)*data(23) + 2*data(8)*data(18)*data(23) + 2*data(3)*data(12)*data(24) + 6*data(6)*data(15)*data(24) - 2*data(7)*data(16)*data(24) - 2*data(8)*data(17)*data(24) + 2*data(9)*data(13)*data(25) - 2*data(7)*data(15)*data(25) - 2*data(6)*data(16)*data(25) + 2*data(4)*data(18)*data(25) + 2*data(9)*data(14)*data(26) - 2*data(8)*data(15)*data(26) - 2*data(6)*data(17)*data(26) + 2*data(5)*data(18)*data(26) + 2*data(7)*data(13)*data(27) + 2*data(8)*data(14)*data(27) + 2*data(4)*data(16)*data(27) + 2*data(5)*data(17)*data(27);
coeffs(168) = data(6)*data(21)^2 + 2*data(3)*data(21)*data(24) + 3*data(6)*data(24)^2 + 2*data(9)*data(22)*data(25) - 2*data(7)*data(24)*data(25) - data(6)*data(25)^2 + 2*data(9)*data(23)*data(26) - 2*data(8)*data(24)*data(26) - data(6)*data(26)^2 + 2*data(7)*data(22)*data(27) + 2*data(8)*data(23)*data(27) + 2*data(4)*data(25)*data(27) + 2*data(5)*data(26)*data(27);
coeffs(169) = data(9)^2*data(15) + 2*data(6)*data(9)*data(18);
coeffs(170) = data(9)^2*data(24) + 2*data(6)*data(9)*data(27);
coeffs(171) = -data(6)*data(10)^2 - data(6)*data(11)^2 + 2*data(4)*data(10)*data(12) + 2*data(5)*data(11)*data(12) + 2*data(3)*data(10)*data(13) + 2*data(1)*data(12)*data(13) + data(6)*data(13)^2 + 2*data(3)*data(11)*data(14) + 2*data(2)*data(12)*data(14) + data(6)*data(14)^2 - 2*data(1)*data(10)*data(15) - 2*data(2)*data(11)*data(15) + 2*data(4)*data(13)*data(15) + 2*data(5)*data(14)*data(15);
coeffs(172) = -2*data(6)*data(10)*data(19) + 2*data(4)*data(12)*data(19) + 2*data(3)*data(13)*data(19) - 2*data(1)*data(15)*data(19) - 2*data(6)*data(11)*data(20) + 2*data(5)*data(12)*data(20) + 2*data(3)*data(14)*data(20) - 2*data(2)*data(15)*data(20) + 2*data(4)*data(10)*data(21) + 2*data(5)*data(11)*data(21) + 2*data(1)*data(13)*data(21) + 2*data(2)*data(14)*data(21) + 2*data(3)*data(10)*data(22) + 2*data(1)*data(12)*data(22) + 2*data(6)*data(13)*data(22) + 2*data(4)*data(15)*data(22) + 2*data(3)*data(11)*data(23) + 2*data(2)*data(12)*data(23) + 2*data(6)*data(14)*data(23) + 2*data(5)*data(15)*data(23) - 2*data(1)*data(10)*data(24) - 2*data(2)*data(11)*data(24) + 2*data(4)*data(13)*data(24) + 2*data(5)*data(14)*data(24);
coeffs(173) = -data(6)*data(19)^2 - data(6)*data(20)^2 + 2*data(4)*data(19)*data(21) + 2*data(5)*data(20)*data(21) + 2*data(3)*data(19)*data(22) + 2*data(1)*data(21)*data(22) + data(6)*data(22)^2 + 2*data(3)*data(20)*data(23) + 2*data(2)*data(21)*data(23) + data(6)*data(23)^2 - 2*data(1)*data(19)*data(24) - 2*data(2)*data(20)*data(24) + 2*data(4)*data(22)*data(24) + 2*data(5)*data(23)*data(24);
coeffs(174) = 2*data(3)*data(6)*data(12) + 2*data(7)*data(9)*data(13) + 2*data(8)*data(9)*data(14) + data(3)^2*data(15) + 3*data(6)^2*data(15) - data(7)^2*data(15) - data(8)^2*data(15) - 2*data(6)*data(7)*data(16) + 2*data(4)*data(9)*data(16) - 2*data(6)*data(8)*data(17) + 2*data(5)*data(9)*data(17) + 2*data(4)*data(7)*data(18) + 2*data(5)*data(8)*data(18);
coeffs(175) = 2*data(3)*data(6)*data(21) + 2*data(7)*data(9)*data(22) + 2*data(8)*data(9)*data(23) + data(3)^2*data(24) + 3*data(6)^2*data(24) - data(7)^2*data(24) - data(8)^2*data(24) - 2*data(6)*data(7)*data(25) + 2*data(4)*data(9)*data(25) - 2*data(6)*data(8)*data(26) + 2*data(5)*data(9)*data(26) + 2*data(4)*data(7)*data(27) + 2*data(5)*data(8)*data(27);
coeffs(176) = data(6)*data(9)^2;
coeffs(177) = 2*data(3)*data(4)*data(10) - 2*data(1)*data(6)*data(10) + 2*data(3)*data(5)*data(11) - 2*data(2)*data(6)*data(11) + 2*data(1)*data(4)*data(12) + 2*data(2)*data(5)*data(12) + 2*data(1)*data(3)*data(13) + 2*data(4)*data(6)*data(13) + 2*data(2)*data(3)*data(14) + 2*data(5)*data(6)*data(14) - data(1)^2*data(15) - data(2)^2*data(15) + data(4)^2*data(15) + data(5)^2*data(15);
coeffs(178) = 2*data(3)*data(4)*data(19) - 2*data(1)*data(6)*data(19) + 2*data(3)*data(5)*data(20) - 2*data(2)*data(6)*data(20) + 2*data(1)*data(4)*data(21) + 2*data(2)*data(5)*data(21) + 2*data(1)*data(3)*data(22) + 2*data(4)*data(6)*data(22) + 2*data(2)*data(3)*data(23) + 2*data(5)*data(6)*data(23) - data(1)^2*data(24) - data(2)^2*data(24) + data(4)^2*data(24) + data(5)^2*data(24);
coeffs(179) = data(3)^2*data(6) + data(6)^3 - data(6)*data(7)^2 - data(6)*data(8)^2 + 2*data(4)*data(7)*data(9) + 2*data(5)*data(8)*data(9);
coeffs(180) = 2*data(1)*data(3)*data(4) + 2*data(2)*data(3)*data(5) - data(1)^2*data(6) - data(2)^2*data(6) + data(4)^2*data(6) + data(5)^2*data(6);
coeffs(181) = data(16)*data(18)^2;
coeffs(182) = data(18)^2*data(25) + 2*data(16)*data(18)*data(27);
coeffs(183) = 2*data(18)*data(25)*data(27) + data(16)*data(27)^2;
coeffs(184) = data(25)*data(27)^2;
coeffs(185) = -data(12)^2*data(16) - data(15)^2*data(16) + data(16)^3 + data(16)*data(17)^2 + 2*data(10)*data(12)*data(18) + 2*data(13)*data(15)*data(18);
coeffs(186) = 2*data(12)*data(18)*data(19) - 2*data(12)*data(16)*data(21) + 2*data(10)*data(18)*data(21) + 2*data(15)*data(18)*data(22) - 2*data(15)*data(16)*data(24) + 2*data(13)*data(18)*data(24) - data(12)^2*data(25) - data(15)^2*data(25) + 3*data(16)^2*data(25) + data(17)^2*data(25) + 2*data(16)*data(17)*data(26) + 2*data(10)*data(12)*data(27) + 2*data(13)*data(15)*data(27);
coeffs(187) = 2*data(18)*data(19)*data(21) - data(16)*data(21)^2 + 2*data(18)*data(22)*data(24) - data(16)*data(24)^2 - 2*data(12)*data(21)*data(25) - 2*data(15)*data(24)*data(25) + 3*data(16)*data(25)^2 + 2*data(17)*data(25)*data(26) + data(16)*data(26)^2 + 2*data(12)*data(19)*data(27) + 2*data(10)*data(21)*data(27) + 2*data(15)*data(22)*data(27) + 2*data(13)*data(24)*data(27);
coeffs(188) = -data(21)^2*data(25) - data(24)^2*data(25) + data(25)^3 + data(25)*data(26)^2 + 2*data(19)*data(21)*data(27) + 2*data(22)*data(24)*data(27);
coeffs(189) = 2*data(9)*data(16)*data(18) + data(7)*data(18)^2;
coeffs(190) = 2*data(9)*data(18)*data(25) + 2*data(9)*data(16)*data(27) + 2*data(7)*data(18)*data(27);
coeffs(191) = 2*data(9)*data(25)*data(27) + data(7)*data(27)^2;
coeffs(192) = data(10)^2*data(16) - data(11)^2*data(16) + data(13)^2*data(16) - data(14)^2*data(16) + 2*data(10)*data(11)*data(17) + 2*data(13)*data(14)*data(17);
coeffs(193) = 2*data(10)*data(16)*data(19) + 2*data(11)*data(17)*data(19) - 2*data(11)*data(16)*data(20) + 2*data(10)*data(17)*data(20) + 2*data(13)*data(16)*data(22) + 2*data(14)*data(17)*data(22) - 2*data(14)*data(16)*data(23) + 2*data(13)*data(17)*data(23) + data(10)^2*data(25) - data(11)^2*data(25) + data(13)^2*data(25) - data(14)^2*data(25) + 2*data(10)*data(11)*data(26) + 2*data(13)*data(14)*data(26);
coeffs(194) = data(16)*data(19)^2 + 2*data(17)*data(19)*data(20) - data(16)*data(20)^2 + data(16)*data(22)^2 + 2*data(17)*data(22)*data(23) - data(16)*data(23)^2 + 2*data(10)*data(19)*data(25) - 2*data(11)*data(20)*data(25) + 2*data(13)*data(22)*data(25) - 2*data(14)*data(23)*data(25) + 2*data(11)*data(19)*data(26) + 2*data(10)*data(20)*data(26) + 2*data(14)*data(22)*data(26) + 2*data(13)*data(23)*data(26);
coeffs(195) = data(19)^2*data(25) - data(20)^2*data(25) + data(22)^2*data(25) - data(23)^2*data(25) + 2*data(19)*data(20)*data(26) + 2*data(22)*data(23)*data(26);
coeffs(196) = 2*data(9)*data(10)*data(12) - data(7)*data(12)^2 + 2*data(9)*data(13)*data(15) - data(7)*data(15)^2 - 2*data(3)*data(12)*data(16) - 2*data(6)*data(15)*data(16) + 3*data(7)*data(16)^2 + 2*data(8)*data(16)*data(17) + data(7)*data(17)^2 + 2*data(3)*data(10)*data(18) + 2*data(1)*data(12)*data(18) + 2*data(6)*data(13)*data(18) + 2*data(4)*data(15)*data(18);
coeffs(197) = 2*data(9)*data(12)*data(19) + 2*data(3)*data(18)*data(19) + 2*data(9)*data(10)*data(21) - 2*data(7)*data(12)*data(21) - 2*data(3)*data(16)*data(21) + 2*data(1)*data(18)*data(21) + 2*data(9)*data(15)*data(22) + 2*data(6)*data(18)*data(22) + 2*data(9)*data(13)*data(24) - 2*data(7)*data(15)*data(24) - 2*data(6)*data(16)*data(24) + 2*data(4)*data(18)*data(24) - 2*data(3)*data(12)*data(25) - 2*data(6)*data(15)*data(25) + 6*data(7)*data(16)*data(25) + 2*data(8)*data(17)*data(25) + 2*data(8)*data(16)*data(26) + 2*data(7)*data(17)*data(26) + 2*data(3)*data(10)*data(27) + 2*data(1)*data(12)*data(27) + 2*data(6)*data(13)*data(27) + 2*data(4)*data(15)*data(27);
coeffs(198) = 2*data(9)*data(19)*data(21) - data(7)*data(21)^2 + 2*data(9)*data(22)*data(24) - data(7)*data(24)^2 - 2*data(3)*data(21)*data(25) - 2*data(6)*data(24)*data(25) + 3*data(7)*data(25)^2 + 2*data(8)*data(25)*data(26) + data(7)*data(26)^2 + 2*data(3)*data(19)*data(27) + 2*data(1)*data(21)*data(27) + 2*data(6)*data(22)*data(27) + 2*data(4)*data(24)*data(27);
coeffs(199) = data(9)^2*data(16) + 2*data(7)*data(9)*data(18);
coeffs(200) = data(9)^2*data(25) + 2*data(7)*data(9)*data(27);
coeffs(201) = data(7)*data(10)^2 + 2*data(8)*data(10)*data(11) - data(7)*data(11)^2 + data(7)*data(13)^2 + 2*data(8)*data(13)*data(14) - data(7)*data(14)^2 + 2*data(1)*data(10)*data(16) - 2*data(2)*data(11)*data(16) + 2*data(4)*data(13)*data(16) - 2*data(5)*data(14)*data(16) + 2*data(2)*data(10)*data(17) + 2*data(1)*data(11)*data(17) + 2*data(5)*data(13)*data(17) + 2*data(4)*data(14)*data(17);
coeffs(202) = 2*data(7)*data(10)*data(19) + 2*data(8)*data(11)*data(19) + 2*data(1)*data(16)*data(19) + 2*data(2)*data(17)*data(19) + 2*data(8)*data(10)*data(20) - 2*data(7)*data(11)*data(20) - 2*data(2)*data(16)*data(20) + 2*data(1)*data(17)*data(20) + 2*data(7)*data(13)*data(22) + 2*data(8)*data(14)*data(22) + 2*data(4)*data(16)*data(22) + 2*data(5)*data(17)*data(22) + 2*data(8)*data(13)*data(23) - 2*data(7)*data(14)*data(23) - 2*data(5)*data(16)*data(23) + 2*data(4)*data(17)*data(23) + 2*data(1)*data(10)*data(25) - 2*data(2)*data(11)*data(25) + 2*data(4)*data(13)*data(25) - 2*data(5)*data(14)*data(25) + 2*data(2)*data(10)*data(26) + 2*data(1)*data(11)*data(26) + 2*data(5)*data(13)*data(26) + 2*data(4)*data(14)*data(26);
coeffs(203) = data(7)*data(19)^2 + 2*data(8)*data(19)*data(20) - data(7)*data(20)^2 + data(7)*data(22)^2 + 2*data(8)*data(22)*data(23) - data(7)*data(23)^2 + 2*data(1)*data(19)*data(25) - 2*data(2)*data(20)*data(25) + 2*data(4)*data(22)*data(25) - 2*data(5)*data(23)*data(25) + 2*data(2)*data(19)*data(26) + 2*data(1)*data(20)*data(26) + 2*data(5)*data(22)*data(26) + 2*data(4)*data(23)*data(26);
coeffs(204) = 2*data(3)*data(9)*data(10) - 2*data(3)*data(7)*data(12) + 2*data(1)*data(9)*data(12) + 2*data(6)*data(9)*data(13) - 2*data(6)*data(7)*data(15) + 2*data(4)*data(9)*data(15) - data(3)^2*data(16) - data(6)^2*data(16) + 3*data(7)^2*data(16) + data(8)^2*data(16) + 2*data(7)*data(8)*data(17) + 2*data(1)*data(3)*data(18) + 2*data(4)*data(6)*data(18);
coeffs(205) = 2*data(3)*data(9)*data(19) - 2*data(3)*data(7)*data(21) + 2*data(1)*data(9)*data(21) + 2*data(6)*data(9)*data(22) - 2*data(6)*data(7)*data(24) + 2*data(4)*data(9)*data(24) - data(3)^2*data(25) - data(6)^2*data(25) + 3*data(7)^2*data(25) + data(8)^2*data(25) + 2*data(7)*data(8)*data(26) + 2*data(1)*data(3)*data(27) + 2*data(4)*data(6)*data(27);
coeffs(206) = data(7)*data(9)^2;
coeffs(207) = 2*data(1)*data(7)*data(10) + 2*data(2)*data(8)*data(10) - 2*data(2)*data(7)*data(11) + 2*data(1)*data(8)*data(11) + 2*data(4)*data(7)*data(13) + 2*data(5)*data(8)*data(13) - 2*data(5)*data(7)*data(14) + 2*data(4)*data(8)*data(14) + data(1)^2*data(16) - data(2)^2*data(16) + data(4)^2*data(16) - data(5)^2*data(16) + 2*data(1)*data(2)*data(17) + 2*data(4)*data(5)*data(17);
coeffs(208) = 2*data(1)*data(7)*data(19) + 2*data(2)*data(8)*data(19) - 2*data(2)*data(7)*data(20) + 2*data(1)*data(8)*data(20) + 2*data(4)*data(7)*data(22) + 2*data(5)*data(8)*data(22) - 2*data(5)*data(7)*data(23) + 2*data(4)*data(8)*data(23) + data(1)^2*data(25) - data(2)^2*data(25) + data(4)^2*data(25) - data(5)^2*data(25) + 2*data(1)*data(2)*data(26) + 2*data(4)*data(5)*data(26);
coeffs(209) = -data(3)^2*data(7) - data(6)^2*data(7) + data(7)^3 + data(7)*data(8)^2 + 2*data(1)*data(3)*data(9) + 2*data(4)*data(6)*data(9);
coeffs(210) = data(1)^2*data(7) - data(2)^2*data(7) + data(4)^2*data(7) - data(5)^2*data(7) + 2*data(1)*data(2)*data(8) + 2*data(4)*data(5)*data(8);
coeffs(211) = data(17)*data(18)^2;
coeffs(212) = data(18)^2*data(26) + 2*data(17)*data(18)*data(27);
coeffs(213) = 2*data(18)*data(26)*data(27) + data(17)*data(27)^2;
coeffs(214) = data(26)*data(27)^2;
coeffs(215) = -data(12)^2*data(17) - data(15)^2*data(17) + data(16)^2*data(17) + data(17)^3 + 2*data(11)*data(12)*data(18) + 2*data(14)*data(15)*data(18);
coeffs(216) = 2*data(12)*data(18)*data(20) - 2*data(12)*data(17)*data(21) + 2*data(11)*data(18)*data(21) + 2*data(15)*data(18)*data(23) - 2*data(15)*data(17)*data(24) + 2*data(14)*data(18)*data(24) + 2*data(16)*data(17)*data(25) - data(12)^2*data(26) - data(15)^2*data(26) + data(16)^2*data(26) + 3*data(17)^2*data(26) + 2*data(11)*data(12)*data(27) + 2*data(14)*data(15)*data(27);
coeffs(217) = 2*data(18)*data(20)*data(21) - data(17)*data(21)^2 + 2*data(18)*data(23)*data(24) - data(17)*data(24)^2 + data(17)*data(25)^2 - 2*data(12)*data(21)*data(26) - 2*data(15)*data(24)*data(26) + 2*data(16)*data(25)*data(26) + 3*data(17)*data(26)^2 + 2*data(12)*data(20)*data(27) + 2*data(11)*data(21)*data(27) + 2*data(15)*data(23)*data(27) + 2*data(14)*data(24)*data(27);
coeffs(218) = -data(21)^2*data(26) - data(24)^2*data(26) + data(25)^2*data(26) + data(26)^3 + 2*data(20)*data(21)*data(27) + 2*data(23)*data(24)*data(27);
coeffs(219) = 2*data(9)*data(17)*data(18) + data(8)*data(18)^2;
coeffs(220) = 2*data(9)*data(18)*data(26) + 2*data(9)*data(17)*data(27) + 2*data(8)*data(18)*data(27);
coeffs(221) = 2*data(9)*data(26)*data(27) + data(8)*data(27)^2;
coeffs(222) = 2*data(10)*data(11)*data(16) + 2*data(13)*data(14)*data(16) - data(10)^2*data(17) + data(11)^2*data(17) - data(13)^2*data(17) + data(14)^2*data(17);
coeffs(223) = 2*data(11)*data(16)*data(19) - 2*data(10)*data(17)*data(19) + 2*data(10)*data(16)*data(20) + 2*data(11)*data(17)*data(20) + 2*data(14)*data(16)*data(22) - 2*data(13)*data(17)*data(22) + 2*data(13)*data(16)*data(23) + 2*data(14)*data(17)*data(23) + 2*data(10)*data(11)*data(25) + 2*data(13)*data(14)*data(25) - data(10)^2*data(26) + data(11)^2*data(26) - data(13)^2*data(26) + data(14)^2*data(26);
coeffs(224) = -data(17)*data(19)^2 + 2*data(16)*data(19)*data(20) + data(17)*data(20)^2 - data(17)*data(22)^2 + 2*data(16)*data(22)*data(23) + data(17)*data(23)^2 + 2*data(11)*data(19)*data(25) + 2*data(10)*data(20)*data(25) + 2*data(14)*data(22)*data(25) + 2*data(13)*data(23)*data(25) - 2*data(10)*data(19)*data(26) + 2*data(11)*data(20)*data(26) - 2*data(13)*data(22)*data(26) + 2*data(14)*data(23)*data(26);
coeffs(225) = 2*data(19)*data(20)*data(25) + 2*data(22)*data(23)*data(25) - data(19)^2*data(26) + data(20)^2*data(26) - data(22)^2*data(26) + data(23)^2*data(26);
coeffs(226) = 2*data(9)*data(11)*data(12) - data(8)*data(12)^2 + 2*data(9)*data(14)*data(15) - data(8)*data(15)^2 + data(8)*data(16)^2 - 2*data(3)*data(12)*data(17) - 2*data(6)*data(15)*data(17) + 2*data(7)*data(16)*data(17) + 3*data(8)*data(17)^2 + 2*data(3)*data(11)*data(18) + 2*data(2)*data(12)*data(18) + 2*data(6)*data(14)*data(18) + 2*data(5)*data(15)*data(18);
coeffs(227) = 2*data(9)*data(12)*data(20) + 2*data(3)*data(18)*data(20) + 2*data(9)*data(11)*data(21) - 2*data(8)*data(12)*data(21) - 2*data(3)*data(17)*data(21) + 2*data(2)*data(18)*data(21) + 2*data(9)*data(15)*data(23) + 2*data(6)*data(18)*data(23) + 2*data(9)*data(14)*data(24) - 2*data(8)*data(15)*data(24) - 2*data(6)*data(17)*data(24) + 2*data(5)*data(18)*data(24) + 2*data(8)*data(16)*data(25) + 2*data(7)*data(17)*data(25) - 2*data(3)*data(12)*data(26) - 2*data(6)*data(15)*data(26) + 2*data(7)*data(16)*data(26) + 6*data(8)*data(17)*data(26) + 2*data(3)*data(11)*data(27) + 2*data(2)*data(12)*data(27) + 2*data(6)*data(14)*data(27) + 2*data(5)*data(15)*data(27);
coeffs(228) = 2*data(9)*data(20)*data(21) - data(8)*data(21)^2 + 2*data(9)*data(23)*data(24) - data(8)*data(24)^2 + data(8)*data(25)^2 - 2*data(3)*data(21)*data(26) - 2*data(6)*data(24)*data(26) + 2*data(7)*data(25)*data(26) + 3*data(8)*data(26)^2 + 2*data(3)*data(20)*data(27) + 2*data(2)*data(21)*data(27) + 2*data(6)*data(23)*data(27) + 2*data(5)*data(24)*data(27);
coeffs(229) = data(9)^2*data(17) + 2*data(8)*data(9)*data(18);
coeffs(230) = data(9)^2*data(26) + 2*data(8)*data(9)*data(27);
coeffs(231) = -data(8)*data(10)^2 + 2*data(7)*data(10)*data(11) + data(8)*data(11)^2 - data(8)*data(13)^2 + 2*data(7)*data(13)*data(14) + data(8)*data(14)^2 + 2*data(2)*data(10)*data(16) + 2*data(1)*data(11)*data(16) + 2*data(5)*data(13)*data(16) + 2*data(4)*data(14)*data(16) - 2*data(1)*data(10)*data(17) + 2*data(2)*data(11)*data(17) - 2*data(4)*data(13)*data(17) + 2*data(5)*data(14)*data(17);
coeffs(232) = -2*data(8)*data(10)*data(19) + 2*data(7)*data(11)*data(19) + 2*data(2)*data(16)*data(19) - 2*data(1)*data(17)*data(19) + 2*data(7)*data(10)*data(20) + 2*data(8)*data(11)*data(20) + 2*data(1)*data(16)*data(20) + 2*data(2)*data(17)*data(20) - 2*data(8)*data(13)*data(22) + 2*data(7)*data(14)*data(22) + 2*data(5)*data(16)*data(22) - 2*data(4)*data(17)*data(22) + 2*data(7)*data(13)*data(23) + 2*data(8)*data(14)*data(23) + 2*data(4)*data(16)*data(23) + 2*data(5)*data(17)*data(23) + 2*data(2)*data(10)*data(25) + 2*data(1)*data(11)*data(25) + 2*data(5)*data(13)*data(25) + 2*data(4)*data(14)*data(25) - 2*data(1)*data(10)*data(26) + 2*data(2)*data(11)*data(26) - 2*data(4)*data(13)*data(26) + 2*data(5)*data(14)*data(26);
coeffs(233) = -data(8)*data(19)^2 + 2*data(7)*data(19)*data(20) + data(8)*data(20)^2 - data(8)*data(22)^2 + 2*data(7)*data(22)*data(23) + data(8)*data(23)^2 + 2*data(2)*data(19)*data(25) + 2*data(1)*data(20)*data(25) + 2*data(5)*data(22)*data(25) + 2*data(4)*data(23)*data(25) - 2*data(1)*data(19)*data(26) + 2*data(2)*data(20)*data(26) - 2*data(4)*data(22)*data(26) + 2*data(5)*data(23)*data(26);
coeffs(234) = 2*data(3)*data(9)*data(11) - 2*data(3)*data(8)*data(12) + 2*data(2)*data(9)*data(12) + 2*data(6)*data(9)*data(14) - 2*data(6)*data(8)*data(15) + 2*data(5)*data(9)*data(15) + 2*data(7)*data(8)*data(16) - data(3)^2*data(17) - data(6)^2*data(17) + data(7)^2*data(17) + 3*data(8)^2*data(17) + 2*data(2)*data(3)*data(18) + 2*data(5)*data(6)*data(18);
coeffs(235) = 2*data(3)*data(9)*data(20) - 2*data(3)*data(8)*data(21) + 2*data(2)*data(9)*data(21) + 2*data(6)*data(9)*data(23) - 2*data(6)*data(8)*data(24) + 2*data(5)*data(9)*data(24) + 2*data(7)*data(8)*data(25) - data(3)^2*data(26) - data(6)^2*data(26) + data(7)^2*data(26) + 3*data(8)^2*data(26) + 2*data(2)*data(3)*data(27) + 2*data(5)*data(6)*data(27);
coeffs(236) = data(8)*data(9)^2;
coeffs(237) = 2*data(2)*data(7)*data(10) - 2*data(1)*data(8)*data(10) + 2*data(1)*data(7)*data(11) + 2*data(2)*data(8)*data(11) + 2*data(5)*data(7)*data(13) - 2*data(4)*data(8)*data(13) + 2*data(4)*data(7)*data(14) + 2*data(5)*data(8)*data(14) + 2*data(1)*data(2)*data(16) + 2*data(4)*data(5)*data(16) - data(1)^2*data(17) + data(2)^2*data(17) - data(4)^2*data(17) + data(5)^2*data(17);
coeffs(238) = 2*data(2)*data(7)*data(19) - 2*data(1)*data(8)*data(19) + 2*data(1)*data(7)*data(20) + 2*data(2)*data(8)*data(20) + 2*data(5)*data(7)*data(22) - 2*data(4)*data(8)*data(22) + 2*data(4)*data(7)*data(23) + 2*data(5)*data(8)*data(23) + 2*data(1)*data(2)*data(25) + 2*data(4)*data(5)*data(25) - data(1)^2*data(26) + data(2)^2*data(26) - data(4)^2*data(26) + data(5)^2*data(26);
coeffs(239) = -data(3)^2*data(8) - data(6)^2*data(8) + data(7)^2*data(8) + data(8)^3 + 2*data(2)*data(3)*data(9) + 2*data(5)*data(6)*data(9);
coeffs(240) = 2*data(1)*data(2)*data(7) + 2*data(4)*data(5)*data(7) - data(1)^2*data(8) + data(2)^2*data(8) - data(4)^2*data(8) + data(5)^2*data(8);
coeffs(241) = data(18)^3;
coeffs(242) = 3*data(18)^2*data(27);
coeffs(243) = 3*data(18)*data(27)^2;
coeffs(244) = data(27)^3;
coeffs(245) = data(12)^2*data(18) + data(15)^2*data(18) + data(16)^2*data(18) + data(17)^2*data(18);
coeffs(246) = 2*data(12)*data(18)*data(21) + 2*data(15)*data(18)*data(24) + 2*data(16)*data(18)*data(25) + 2*data(17)*data(18)*data(26) + data(12)^2*data(27) + data(15)^2*data(27) + data(16)^2*data(27) + data(17)^2*data(27);
coeffs(247) = data(18)*data(21)^2 + data(18)*data(24)^2 + data(18)*data(25)^2 + data(18)*data(26)^2 + 2*data(12)*data(21)*data(27) + 2*data(15)*data(24)*data(27) + 2*data(16)*data(25)*data(27) + 2*data(17)*data(26)*data(27);
coeffs(248) = data(21)^2*data(27) + data(24)^2*data(27) + data(25)^2*data(27) + data(26)^2*data(27);
coeffs(249) = 3*data(9)*data(18)^2;
coeffs(250) = 6*data(9)*data(18)*data(27);
coeffs(251) = 3*data(9)*data(27)^2;
coeffs(252) = 2*data(10)*data(12)*data(16) + 2*data(13)*data(15)*data(16) + 2*data(11)*data(12)*data(17) + 2*data(14)*data(15)*data(17) - data(10)^2*data(18) - data(11)^2*data(18) - data(13)^2*data(18) - data(14)^2*data(18);
coeffs(253) = 2*data(12)*data(16)*data(19) - 2*data(10)*data(18)*data(19) + 2*data(12)*data(17)*data(20) - 2*data(11)*data(18)*data(20) + 2*data(10)*data(16)*data(21) + 2*data(11)*data(17)*data(21) + 2*data(15)*data(16)*data(22) - 2*data(13)*data(18)*data(22) + 2*data(15)*data(17)*data(23) - 2*data(14)*data(18)*data(23) + 2*data(13)*data(16)*data(24) + 2*data(14)*data(17)*data(24) + 2*data(10)*data(12)*data(25) + 2*data(13)*data(15)*data(25) + 2*data(11)*data(12)*data(26) + 2*data(14)*data(15)*data(26) - data(10)^2*data(27) - data(11)^2*data(27) - data(13)^2*data(27) - data(14)^2*data(27);
coeffs(254) = -data(18)*data(19)^2 - data(18)*data(20)^2 + 2*data(16)*data(19)*data(21) + 2*data(17)*data(20)*data(21) - data(18)*data(22)^2 - data(18)*data(23)^2 + 2*data(16)*data(22)*data(24) + 2*data(17)*data(23)*data(24) + 2*data(12)*data(19)*data(25) + 2*data(10)*data(21)*data(25) + 2*data(15)*data(22)*data(25) + 2*data(13)*data(24)*data(25) + 2*data(12)*data(20)*data(26) + 2*data(11)*data(21)*data(26) + 2*data(15)*data(23)*data(26) + 2*data(14)*data(24)*data(26) - 2*data(10)*data(19)*data(27) - 2*data(11)*data(20)*data(27) - 2*data(13)*data(22)*data(27) - 2*data(14)*data(23)*data(27);
coeffs(255) = 2*data(19)*data(21)*data(25) + 2*data(22)*data(24)*data(25) + 2*data(20)*data(21)*data(26) + 2*data(23)*data(24)*data(26) - data(19)^2*data(27) - data(20)^2*data(27) - data(22)^2*data(27) - data(23)^2*data(27);
coeffs(256) = data(9)*data(12)^2 + data(9)*data(15)^2 + data(9)*data(16)^2 + data(9)*data(17)^2 + 2*data(3)*data(12)*data(18) + 2*data(6)*data(15)*data(18) + 2*data(7)*data(16)*data(18) + 2*data(8)*data(17)*data(18);
coeffs(257) = 2*data(9)*data(12)*data(21) + 2*data(3)*data(18)*data(21) + 2*data(9)*data(15)*data(24) + 2*data(6)*data(18)*data(24) + 2*data(9)*data(16)*data(25) + 2*data(7)*data(18)*data(25) + 2*data(9)*data(17)*data(26) + 2*data(8)*data(18)*data(26) + 2*data(3)*data(12)*data(27) + 2*data(6)*data(15)*data(27) + 2*data(7)*data(16)*data(27) + 2*data(8)*data(17)*data(27);
coeffs(258) = data(9)*data(21)^2 + data(9)*data(24)^2 + data(9)*data(25)^2 + data(9)*data(26)^2 + 2*data(3)*data(21)*data(27) + 2*data(6)*data(24)*data(27) + 2*data(7)*data(25)*data(27) + 2*data(8)*data(26)*data(27);
coeffs(259) = 3*data(9)^2*data(18);
coeffs(260) = 3*data(9)^2*data(27);
coeffs(261) = -data(9)*data(10)^2 - data(9)*data(11)^2 + 2*data(7)*data(10)*data(12) + 2*data(8)*data(11)*data(12) - data(9)*data(13)^2 - data(9)*data(14)^2 + 2*data(7)*data(13)*data(15) + 2*data(8)*data(14)*data(15) + 2*data(3)*data(10)*data(16) + 2*data(1)*data(12)*data(16) + 2*data(6)*data(13)*data(16) + 2*data(4)*data(15)*data(16) + 2*data(3)*data(11)*data(17) + 2*data(2)*data(12)*data(17) + 2*data(6)*data(14)*data(17) + 2*data(5)*data(15)*data(17) - 2*data(1)*data(10)*data(18) - 2*data(2)*data(11)*data(18) - 2*data(4)*data(13)*data(18) - 2*data(5)*data(14)*data(18);
coeffs(262) = -2*data(9)*data(10)*data(19) + 2*data(7)*data(12)*data(19) + 2*data(3)*data(16)*data(19) - 2*data(1)*data(18)*data(19) - 2*data(9)*data(11)*data(20) + 2*data(8)*data(12)*data(20) + 2*data(3)*data(17)*data(20) - 2*data(2)*data(18)*data(20) + 2*data(7)*data(10)*data(21) + 2*data(8)*data(11)*data(21) + 2*data(1)*data(16)*data(21) + 2*data(2)*data(17)*data(21) - 2*data(9)*data(13)*data(22) + 2*data(7)*data(15)*data(22) + 2*data(6)*data(16)*data(22) - 2*data(4)*data(18)*data(22) - 2*data(9)*data(14)*data(23) + 2*data(8)*data(15)*data(23) + 2*data(6)*data(17)*data(23) - 2*data(5)*data(18)*data(23) + 2*data(7)*data(13)*data(24) + 2*data(8)*data(14)*data(24) + 2*data(4)*data(16)*data(24) + 2*data(5)*data(17)*data(24) + 2*data(3)*data(10)*data(25) + 2*data(1)*data(12)*data(25) + 2*data(6)*data(13)*data(25) + 2*data(4)*data(15)*data(25) + 2*data(3)*data(11)*data(26) + 2*data(2)*data(12)*data(26) + 2*data(6)*data(14)*data(26) + 2*data(5)*data(15)*data(26) - 2*data(1)*data(10)*data(27) - 2*data(2)*data(11)*data(27) - 2*data(4)*data(13)*data(27) - 2*data(5)*data(14)*data(27);
coeffs(263) = -data(9)*data(19)^2 - data(9)*data(20)^2 + 2*data(7)*data(19)*data(21) + 2*data(8)*data(20)*data(21) - data(9)*data(22)^2 - data(9)*data(23)^2 + 2*data(7)*data(22)*data(24) + 2*data(8)*data(23)*data(24) + 2*data(3)*data(19)*data(25) + 2*data(1)*data(21)*data(25) + 2*data(6)*data(22)*data(25) + 2*data(4)*data(24)*data(25) + 2*data(3)*data(20)*data(26) + 2*data(2)*data(21)*data(26) + 2*data(6)*data(23)*data(26) + 2*data(5)*data(24)*data(26) - 2*data(1)*data(19)*data(27) - 2*data(2)*data(20)*data(27) - 2*data(4)*data(22)*data(27) - 2*data(5)*data(23)*data(27);
coeffs(264) = 2*data(3)*data(9)*data(12) + 2*data(6)*data(9)*data(15) + 2*data(7)*data(9)*data(16) + 2*data(8)*data(9)*data(17) + data(3)^2*data(18) + data(6)^2*data(18) + data(7)^2*data(18) + data(8)^2*data(18);
coeffs(265) = 2*data(3)*data(9)*data(21) + 2*data(6)*data(9)*data(24) + 2*data(7)*data(9)*data(25) + 2*data(8)*data(9)*data(26) + data(3)^2*data(27) + data(6)^2*data(27) + data(7)^2*data(27) + data(8)^2*data(27);
coeffs(266) = data(9)^3;
coeffs(267) = 2*data(3)*data(7)*data(10) - 2*data(1)*data(9)*data(10) + 2*data(3)*data(8)*data(11) - 2*data(2)*data(9)*data(11) + 2*data(1)*data(7)*data(12) + 2*data(2)*data(8)*data(12) + 2*data(6)*data(7)*data(13) - 2*data(4)*data(9)*data(13) + 2*data(6)*data(8)*data(14) - 2*data(5)*data(9)*data(14) + 2*data(4)*data(7)*data(15) + 2*data(5)*data(8)*data(15) + 2*data(1)*data(3)*data(16) + 2*data(4)*data(6)*data(16) + 2*data(2)*data(3)*data(17) + 2*data(5)*data(6)*data(17) - data(1)^2*data(18) - data(2)^2*data(18) - data(4)^2*data(18) - data(5)^2*data(18);
coeffs(268) = 2*data(3)*data(7)*data(19) - 2*data(1)*data(9)*data(19) + 2*data(3)*data(8)*data(20) - 2*data(2)*data(9)*data(20) + 2*data(1)*data(7)*data(21) + 2*data(2)*data(8)*data(21) + 2*data(6)*data(7)*data(22) - 2*data(4)*data(9)*data(22) + 2*data(6)*data(8)*data(23) - 2*data(5)*data(9)*data(23) + 2*data(4)*data(7)*data(24) + 2*data(5)*data(8)*data(24) + 2*data(1)*data(3)*data(25) + 2*data(4)*data(6)*data(25) + 2*data(2)*data(3)*data(26) + 2*data(5)*data(6)*data(26) - data(1)^2*data(27) - data(2)^2*data(27) - data(4)^2*data(27) - data(5)^2*data(27);
coeffs(269) = data(3)^2*data(9) + data(6)^2*data(9) + data(7)^2*data(9) + data(8)^2*data(9);
coeffs(270) = 2*data(1)*data(3)*data(7) + 2*data(4)*data(6)*data(7) + 2*data(2)*data(3)*data(8) + 2*data(5)*data(6)*data(8) - data(1)^2*data(9) - data(2)^2*data(9) - data(4)^2*data(9) - data(5)^2*data(9);
coeffs(271) = -data(12)*data(14)*data(16) + data(11)*data(15)*data(16) + data(12)*data(13)*data(17) - data(10)*data(15)*data(17) - data(11)*data(13)*data(18) + data(10)*data(14)*data(18);
coeffs(272) = -data(15)*data(17)*data(19) + data(14)*data(18)*data(19) + data(15)*data(16)*data(20) - data(13)*data(18)*data(20) - data(14)*data(16)*data(21) + data(13)*data(17)*data(21) + data(12)*data(17)*data(22) - data(11)*data(18)*data(22) - data(12)*data(16)*data(23) + data(10)*data(18)*data(23) + data(11)*data(16)*data(24) - data(10)*data(17)*data(24) - data(12)*data(14)*data(25) + data(11)*data(15)*data(25) + data(12)*data(13)*data(26) - data(10)*data(15)*data(26) - data(11)*data(13)*data(27) + data(10)*data(14)*data(27);
coeffs(273) = -data(18)*data(20)*data(22) + data(17)*data(21)*data(22) + data(18)*data(19)*data(23) - data(16)*data(21)*data(23) - data(17)*data(19)*data(24) + data(16)*data(20)*data(24) + data(15)*data(20)*data(25) - data(14)*data(21)*data(25) - data(12)*data(23)*data(25) + data(11)*data(24)*data(25) - data(15)*data(19)*data(26) + data(13)*data(21)*data(26) + data(12)*data(22)*data(26) - data(10)*data(24)*data(26) + data(14)*data(19)*data(27) - data(13)*data(20)*data(27) - data(11)*data(22)*data(27) + data(10)*data(23)*data(27);
coeffs(274) = -data(21)*data(23)*data(25) + data(20)*data(24)*data(25) + data(21)*data(22)*data(26) - data(19)*data(24)*data(26) - data(20)*data(22)*data(27) + data(19)*data(23)*data(27);
coeffs(275) = -data(9)*data(11)*data(13) + data(8)*data(12)*data(13) + data(9)*data(10)*data(14) - data(7)*data(12)*data(14) - data(8)*data(10)*data(15) + data(7)*data(11)*data(15) + data(6)*data(11)*data(16) - data(5)*data(12)*data(16) - data(3)*data(14)*data(16) + data(2)*data(15)*data(16) - data(6)*data(10)*data(17) + data(4)*data(12)*data(17) + data(3)*data(13)*data(17) - data(1)*data(15)*data(17) + data(5)*data(10)*data(18) - data(4)*data(11)*data(18) - data(2)*data(13)*data(18) + data(1)*data(14)*data(18);
coeffs(276) = data(9)*data(14)*data(19) - data(8)*data(15)*data(19) - data(6)*data(17)*data(19) + data(5)*data(18)*data(19) - data(9)*data(13)*data(20) + data(7)*data(15)*data(20) + data(6)*data(16)*data(20) - data(4)*data(18)*data(20) + data(8)*data(13)*data(21) - data(7)*data(14)*data(21) - data(5)*data(16)*data(21) + data(4)*data(17)*data(21) - data(9)*data(11)*data(22) + data(8)*data(12)*data(22) + data(3)*data(17)*data(22) - data(2)*data(18)*data(22) + data(9)*data(10)*data(23) - data(7)*data(12)*data(23) - data(3)*data(16)*data(23) + data(1)*data(18)*data(23) - data(8)*data(10)*data(24) + data(7)*data(11)*data(24) + data(2)*data(16)*data(24) - data(1)*data(17)*data(24) + data(6)*data(11)*data(25) - data(5)*data(12)*data(25) - data(3)*data(14)*data(25) + data(2)*data(15)*data(25) - data(6)*data(10)*data(26) + data(4)*data(12)*data(26) + data(3)*data(13)*data(26) - data(1)*data(15)*data(26) + data(5)*data(10)*data(27) - data(4)*data(11)*data(27) - data(2)*data(13)*data(27) + data(1)*data(14)*data(27);
coeffs(277) = -data(9)*data(20)*data(22) + data(8)*data(21)*data(22) + data(9)*data(19)*data(23) - data(7)*data(21)*data(23) - data(8)*data(19)*data(24) + data(7)*data(20)*data(24) + data(6)*data(20)*data(25) - data(5)*data(21)*data(25) - data(3)*data(23)*data(25) + data(2)*data(24)*data(25) - data(6)*data(19)*data(26) + data(4)*data(21)*data(26) + data(3)*data(22)*data(26) - data(1)*data(24)*data(26) + data(5)*data(19)*data(27) - data(4)*data(20)*data(27) - data(2)*data(22)*data(27) + data(1)*data(23)*data(27);
coeffs(278) = -data(6)*data(8)*data(10) + data(5)*data(9)*data(10) + data(6)*data(7)*data(11) - data(4)*data(9)*data(11) - data(5)*data(7)*data(12) + data(4)*data(8)*data(12) + data(3)*data(8)*data(13) - data(2)*data(9)*data(13) - data(3)*data(7)*data(14) + data(1)*data(9)*data(14) + data(2)*data(7)*data(15) - data(1)*data(8)*data(15) - data(3)*data(5)*data(16) + data(2)*data(6)*data(16) + data(3)*data(4)*data(17) - data(1)*data(6)*data(17) - data(2)*data(4)*data(18) + data(1)*data(5)*data(18);
coeffs(279) = -data(6)*data(8)*data(19) + data(5)*data(9)*data(19) + data(6)*data(7)*data(20) - data(4)*data(9)*data(20) - data(5)*data(7)*data(21) + data(4)*data(8)*data(21) + data(3)*data(8)*data(22) - data(2)*data(9)*data(22) - data(3)*data(7)*data(23) + data(1)*data(9)*data(23) + data(2)*data(7)*data(24) - data(1)*data(8)*data(24) - data(3)*data(5)*data(25) + data(2)*data(6)*data(25) + data(3)*data(4)*data(26) - data(1)*data(6)*data(26) - data(2)*data(4)*data(27) + data(1)*data(5)*data(27);
coeffs(280) = -data(3)*data(5)*data(7) + data(2)*data(6)*data(7) + data(3)*data(4)*data(8) - data(1)*data(6)*data(8) - data(2)*data(4)*data(9) + data(1)*data(5)*data(9);
end
function [C0,C1] = setup_elimination_template(data)
[coeffs] = compute_coeffs(data);
coeffs0_ind = [1,31,61,91,121,151,181,211,241,2,32,62,92,122,152,182,212,242,3,33,63,93,123,153,183,213,243,5,35,65,31,1,61,95,125,155,91,121,151,185,215,181,211,241,245,271,6,36,66,32,...
2,62,96,126,156,92,122,152,186,216,182,212,242,246,272,7,37,67,33,3,63,97,127,157,93,123,153,187,217,183,213,243,247,273,8,38,68,34,4,64,98,128,158,94,124,154,188,218,184,214,...
244,248,274,9,39,69,99,129,159,189,219,249,10,40,70,100,130,160,190,220,250,12,42,72,35,5,65,102,132,162,91,95,61,121,125,155,192,222,1,181,185,151,211,215,31,241,245,252,271,13,...
43,73,36,6,66,103,133,163,92,96,62,122,126,156,193,223,2,182,186,152,212,216,32,242,246,253,272,14,44,74,37,7,67,104,134,164,93,97,63,123,127,157,194,224,3,183,187,153,213,217,...
33,243,247,254,273,15,45,75,38,8,68,105,135,165,94,98,64,124,128,158,195,225,4,184,188,154,214,218,34,244,248,255,274,16,46,76,39,9,69,106,136,166,99,129,159,196,226,189,219,249,...
256,275,17,47,77,40,10,70,107,137,167,100,130,160,197,227,190,220,250,257,276,18,48,78,41,11,71,108,138,168,101,131,161,198,228,191,221,251,258,277,19,49,79,109,139,169,199,229,259,42,...
12,72,95,102,65,125,132,162,5,185,192,155,215,222,35,245,252,271,43,13,73,96,103,66,126,133,163,6,186,193,156,216,223,36,246,253,272,21,51,81,46,16,76,111,141,171,99,106,69,129,...
136,166,201,231,9,189,196,159,219,226,39,249,256,261,275,24,54,84,49,19,79,114,144,174,109,139,169,204,234,199,229,259,264,278,102,72,132,12,192,162,222,42,252,271,51,21,81,106,111,76,...
136,141,171,16,196,201,166,226,231,46,256,261,275,103,73,133,13,193,163,223,43,253,272,104,74,134,14,194,164,224,44,254,273,44,14,74,97,104,67,127,134,164,7,187,194,157,217,224,37,247,...
254,273,22,52,82,47,17,77,112,142,172,100,107,70,130,137,167,202,232,10,190,197,160,220,227,40,250,257,262,276,105,75,135,15,195,165,225,45,255,274,45,15,75,98,105,68,128,135,165,8,...
188,195,158,218,225,38,248,255,274,23,53,83,48,18,78,113,143,173,101,108,71,131,138,168,203,233,11,191,198,161,221,228,41,251,258,263,277,25,55,85,50,20,80,115,145,175,110,140,170,205,...
235,200,230,260,265,279];
coeffs1_ind = [120,90,150,30,210,180,240,60,270,280,117,87,147,27,207,177,237,57,267,278,111,81,141,21,201,171,231,51,261,275,112,82,142,22,202,172,232,52,262,276,52,22,82,107,112,77,137,142,172,17,...
197,202,167,227,232,47,257,262,276,57,27,87,114,117,84,144,147,177,24,204,207,174,234,237,54,264,267,278,27,57,87,54,24,84,117,147,177,109,114,79,139,144,174,207,237,19,199,204,169,229,...
234,49,259,264,267,278,118,88,148,28,208,178,238,58,268,279,113,83,143,23,203,173,233,53,263,277,53,23,83,108,113,78,138,143,173,18,198,203,168,228,233,48,258,263,277,58,28,88,115,118,...
85,145,148,178,25,205,208,175,235,238,55,265,268,279,28,58,88,55,25,85,118,148,178,110,115,80,140,145,175,208,238,20,200,205,170,230,235,50,260,265,268,279,60,30,90,119,120,89,149,150,...
180,29,209,210,179,239,240,59,269,270,280,30,60,90,59,29,89,120,150,180,116,119,86,146,149,179,210,240,26,206,209,176,236,239,56,266,269,270,280,29,59,89,56,26,86,119,149,179,116,146,...
176,209,239,206,236,266,269,280];
C0_ind = [1,2,3,7,8,9,16,17,27,32,33,34,38,39,40,47,48,58,63,64,65,69,70,71,78,79,89,94,95,96,97,98,99,100,101,102,104,107,108,109,110,113,116,119,120,124,125,126,127,128,...
129,130,131,132,133,135,138,139,140,141,144,147,150,151,155,156,157,158,159,160,161,162,163,164,166,169,170,171,172,175,178,181,182,186,187,188,189,190,191,192,193,194,195,197,200,201,202,203,206,209,...
212,213,217,218,219,220,224,225,226,233,234,244,249,250,251,255,256,257,264,265,275,280,281,282,283,284,285,286,287,288,289,290,291,292,293,294,295,296,297,298,299,300,301,302,303,304,305,306,309,311,...
312,313,314,315,316,317,318,319,320,321,322,323,324,325,326,327,328,329,330,331,332,333,334,335,336,337,340,342,343,344,345,346,347,348,349,350,351,352,353,354,355,356,357,358,359,360,361,362,363,364,...
365,366,367,368,371,373,374,375,376,377,378,379,380,381,382,383,384,385,386,387,388,389,390,391,392,393,394,395,396,397,398,399,402,404,405,406,407,408,409,410,411,412,414,417,418,419,420,423,426,429,...
430,434,435,436,437,438,439,440,441,442,443,445,448,449,450,451,454,457,460,461,465,466,467,468,469,470,471,472,473,474,476,479,480,481,482,485,488,491,492,496,497,498,499,503,504,505,512,513,523,531,...
532,533,537,538,539,540,541,542,545,546,547,548,549,550,551,552,553,556,562,563,564,568,569,570,571,572,573,576,577,578,579,580,581,582,583,584,587,590,591,592,593,594,595,596,597,598,599,600,601,602,...
603,604,605,606,607,608,609,610,611,612,613,614,615,616,619,621,622,623,624,625,626,627,628,629,631,634,635,636,637,640,643,646,647,651,661,663,664,669,670,672,673,675,676,679,686,687,688,692,693,694,...
695,696,697,700,701,702,703,704,705,706,707,708,711,723,725,726,731,732,734,735,737,738,741,754,756,757,762,763,765,766,768,769,772,779,780,781,785,786,787,788,789,790,793,794,795,796,797,798,799,800,...
801,804,807,808,809,810,811,812,813,814,815,816,817,818,819,820,821,822,823,824,825,826,827,828,829,830,831,832,833,836,847,849,850,855,856,858,859,861,862,865,872,873,874,878,879,880,881,882,883,886,...
887,888,889,890,891,892,893,894,897,900,901,902,903,904,905,906,907,908,909,910,911,912,913,914,915,916,917,918,919,920,921,922,923,924,925,926,929,931,932,933,934,935,936,937,938,939,941,944,945,946,...
947,950,953,956,957,961];
C1_ind = [10,12,13,18,19,21,22,24,25,28,41,43,44,49,50,52,53,55,56,59,72,74,75,80,81,83,84,86,87,90,103,105,106,111,112,114,115,117,118,121,128,129,130,134,135,136,137,138,139,142,...
143,144,145,146,147,148,149,150,153,159,160,161,165,166,167,168,169,170,173,174,175,176,177,178,179,180,181,184,187,188,189,190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,...
209,210,211,212,213,216,227,229,230,235,236,238,239,241,242,245,258,260,261,266,267,269,270,272,273,276,283,284,285,289,290,291,292,293,294,297,298,299,300,301,302,303,304,305,308,314,315,316,320,321,...
322,323,324,325,328,329,330,331,332,333,334,335,336,339,342,343,344,345,346,347,348,349,350,351,352,353,354,355,356,357,358,359,360,361,362,363,364,365,366,367,368,371,376,377,378,382,383,384,385,386,...
387,390,391,392,393,394,395,396,397,398,401,404,405,406,407,408,409,410,411,412,413,414,415,416,417,418,419,420,421,422,423,424,425,426,427,428,429,430,433,435,436,437,438,439,440,441,442,443,445,448,...
449,450,451,454,457,460,461,465];
C0 = zeros(31,31);
C1 = zeros(31,15);
C0(C0_ind) = coeffs(coeffs0_ind);
C1(C1_ind) = coeffs(coeffs1_ind);
end