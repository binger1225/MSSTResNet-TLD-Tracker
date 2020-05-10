function imdb = cnn_setup_imdb(imgsCell, labelsCell)


% construct imdb
imdb = struct();
imdb.images = struct();
imdb.labels = struct();

cnt = 0;
for j = 1:numel(imgsCell)
    imgs = imgsCell{j};
    labels = labelsCell{j};
    for k = 1: size(imgs,4)
        cnt = cnt + 1;
        imdb.images.img{cnt} = imgs(:,:,:,k);
        imdb.images.set(cnt) = 1;
        rects = double(labels(:,:,k));
        imdb.labels.rects{cnt} = horzcat(...
            rects(:,[2 1]), rects(:,[2 1])+rects(:,[4 3])-1);
        imdb.labels.eventid(cnt) = 1;
    end
end


end

