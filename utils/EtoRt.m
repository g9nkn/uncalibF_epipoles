% Copyright (c) 2023 NEC Corporation
% EtoRt decomposes an essential matrix into relative rotation and translation
% 
% usage: [R, t]= EtoRt(E)
% 
% arguments:
%	E - 3x3 an essential matrix such that E = [t]xR and m2'*E*m1=0
% returns:
%	R - 3x3x4 rotation matrix
%	t - 3x4 translation vector
%		--> R(:,:,i) and t(:,i) satisfy E = [t(:,i)]xR(:,:,i)
%
% references:
%	B. K. P. Horn, "Recovering baseline and orientation from essential matrix," J. Optical Society of America, 1990.
function [R, t]= EtoRt(E)
	
	E = E./norm(E,'fro')*sqrt(2);
	
	% translation
	e1 = E(:,1);
	e2 = E(:,2);
	e3 = E(:,3);
	
	e1xe2 = cross(e1,e2);
	e1xe3 = cross(e1,e3);
	e2xe3 = cross(e2,e3);
	
	norm1 = norm(e1xe2);
	norm2 = norm(e1xe3);
	norm3 = norm(e2xe3);
	
	[~, idx] = max([norm1, norm2, norm3]);
	
    switch idx
        case 1
            v = e1xe2;
        case 2
            v = e1xe3;
        case 3
            v = e2xe3;
    end
    t1 = v/norm(v);
    t2 = -t1;
	
	
	% rotation
	CofEt = [e2xe3, -e1xe3, e1xe2]; % transposed Cofactors of E
	R1 = CofEt - skew3x3(t1)*E;
	R2 = CofEt - skew3x3(t2)*E;
	
	% output
	R = cat(3, R1, R1, R2, R2);	
	t = [t1,t2,t1,t2];
return