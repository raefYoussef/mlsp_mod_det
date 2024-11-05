function [const] = GenConst(sigIQ)
%GenConst Generate Constellation From Signal IQ
%
% Inputs:
%   sigIQ           - Signal IQ
% Outputs:
%   const           - Constellation Matrix

% normalize IQ data
maxRange = 2; % Define max range for the constellation
sigIQ = maxRange * sigIQ / max(abs(sigIQ)); % Normalize within specified range

% define grid resolution for pixel positions
gridSize = 100;
xEdges = linspace(-maxRange, maxRange, gridSize+1);
yEdges = linspace(-maxRange, maxRange, gridSize+1);

% generate 2D histogram (constellation as pixel data)
const = histcounts2(real(sigIQ), imag(sigIQ), xEdges, yEdges).';

% normalize constellation
const = const ./ max(const, [], 'all');

end