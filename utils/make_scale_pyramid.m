% -----------------------------------------------------------------------------------------------------
function pyramid = make_scale_pyramid(im, targetPosition, in_side_scaled, out_side, avgChans, p)
%MAKE_SCALE_PYRAMID
%   computes a pyramid of re-scaled copies of the target (centered on TARGETPOSITION)
%   and resizes them to OUT_SIDE. If crops exceed image boundaries they are padded with AVGCHANS.
%
%   Luca Bertinetto, Jack Valmadre, Joao F. Henriques, 2016
% -----------------------------------------------------------------------------------------------------
    in_side_scaled = round(in_side_scaled);
%     pyramid = gpuArray(zeros(out_side, out_side, 3, p.numScale, 'single'));
    pyramid = zeros(out_side, out_side, 3, p.numScale, 'single');
    max_target_side = in_side_scaled(end,:);
    min_target_side = in_side_scaled(1,:);
    beta = [out_side out_side] ./ min_target_side;
    search_side = round(beta .* max_target_side);
    [search_region, ~] = get_subwindow_tracking(im, targetPosition, search_side, max_target_side, avgChans);
    if p.subMean
        search_region = bsxfun(@minus, search_region, reshape(avgChans, [1 1 3]));
    end

    for s = 1:p.numScale
        target_side = round(beta .* in_side_scaled(s,:));
        [pyramid(:,:,:,s), ~] = get_subwindow_tracking(search_region, (1+search_side)/2, [out_side out_side], target_side, avgChans);
    end
end