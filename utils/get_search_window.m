function window_sz = get_search_window( target_sz, im_sz, opts)
% GET_SEARCH_WINDOW
    contextAmount = 0.5;
    wc_z = target_sz(2) + contextAmount * sum(target_sz);
    hc_z = target_sz(1) + contextAmount * sum(target_sz);
    s_z = sqrt(wc_z*hc_z);
    window_sz = round([s_z s_z]);
end

