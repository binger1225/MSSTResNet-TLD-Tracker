% --------------------------------------------------------------------------------------------------------
function [im_patch, im_patch_original, label_crop] = get_subwindow(im, pos, target_sz, model_sz, original_sz, avg_chans)
%GET_SUBWINDOW_TRACKING Obtain image sub-window, padding with avg channel if area goes outside of border
% -------------------------------------------------------------------------------------------------
    if isempty(original_sz)
        original_sz = model_sz;
    end
    sz = original_sz;
    im_sz = size(im);
    %make sure the size is not too small
    assert(all(im_sz(1:2) > 2));
    c = (1+sz) / 2;
    context_xmin = round(pos(2) - c(2));
    context_ymin = round(pos(1) - c(1));
    context_xmax = context_xmin + sz(2) - 1;
    context_ymax = context_ymin + sz(1) - 1;
    left_pad = max(0, 1-context_xmin);
    top_pad = max(0, 1-context_ymin);
    right_pad = max(0, context_xmax - im_sz(2));
    bottom_pad = max(0, context_ymax - im_sz(1));

    context_xmin = context_xmin + left_pad;
    context_xmax = context_xmax + left_pad;
    context_ymin = context_ymin + top_pad;
    context_ymax = context_ymax + top_pad;
    
    if top_pad || left_pad
        R = padarray(im(:,:,1), [top_pad left_pad], avg_chans(1), 'pre');
        G = padarray(im(:,:,2), [top_pad left_pad], avg_chans(2), 'pre');
        B = padarray(im(:,:,3), [top_pad left_pad], avg_chans(3), 'pre');
        im = cat(3, R, G, B);
    end

    if bottom_pad || right_pad
        R = padarray(im(:,:,1), [bottom_pad right_pad], avg_chans(1), 'post');
        G = padarray(im(:,:,2), [bottom_pad right_pad], avg_chans(2), 'post');
        B = padarray(im(:,:,3), [bottom_pad right_pad], avg_chans(3), 'post');
        im = cat(3, R, G, B);
    end

    xs = context_xmin : context_xmax;
    ys = context_ymin : context_ymax;

    im_patch_original = im(ys, xs, :);
     % center-y center-x height width 
    label_crop = [pos target_sz];
    label_crop([1 2]) = round(pos([1,2]) - (1+target_sz([1,2]))/2 - [context_ymin context_xmin] + [top_pad left_pad]);
    if ~isequal(model_sz, original_sz)
        im_patch = imresize(im_patch_original, model_sz, 'bilinear');
        label_crop  = round(label_crop .* repmat(model_sz./original_sz, 1, 2));
    else
        im_patch = im_patch_original;
    end
end
