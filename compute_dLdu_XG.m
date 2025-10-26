%% ===================================================================== %%
%% Gradient of XG Loss w.r.t. residual u                                  %
%% ===================================================================== %%
function dL_du = compute_dLdu_XG(u, a, eta)
% compute_dLdu_XG
% -------------------------------------------------------------------------
% Compute ∂L_XG(u)/∂u elementwise for the proposed XG loss.
%
% INPUT:
%   u   : [m x K] residual matrix, where u(i,j) = (ZΘ)_ij - Y_ij
%   a   : XG-loss parameter "a" (controls asymmetry / slope)
%   eta : XG-loss parameter "η" (>0) (controls bounded influence)
%
% OUTPUT:
%   dL_du : [m x K] matrix of partial derivatives ∂L_XG/∂u(i,j)
%
%
% Original scalar formula for each residual u_ij was:
%
%   dL/du = [ a * (a*u + 1) * exp(a*u) - a ] / [ 1 + a*eta*u*(exp(a*u)-1) ]^2
%
%
% This derivative drives robustness and is what replaces the standard
% squared loss gradient in RVFL training.
% -------------------------------------------------------------------------

    % Precompute exp(a*u) elementwise
    au      = a .* u;              % a*u
    exp_au  = exp(au);             % exp(a*u)
    % Numerator: a * (a*u + 1) * exp(a*u) - a
    numerator = a .* (au + 1) .* exp_au - a;

    % Denominator base: 1 + a*eta*u*(exp(a*u) - 1)
    denom_base = 1 + (a .* eta .* u .* (exp_au - 1));

    % Square the denominator
    denom = denom_base .^ 2;

    % Final derivative
    dL_du = numerator ./ denom;
end



