function opt = get_opt(box_map,thresh)
%一个box_map即一个填涂框
%此函数通过选项填涂框面积以及面积阈值判断当前box_map是否被填涂
if nargin < 2
    [h,w]=size(box_map);
    thresh = h*w*0.2;%默认阈值为box的面积的1/5
end
area = bwarea(box_map);
if area>thresh%面积大于阈值
    opt=1;
else
    opt=0;
end
end