function opt = get_opt(box_map,thresh)
%һ��box_map��һ����Ϳ��
%�˺���ͨ��ѡ����Ϳ������Լ������ֵ�жϵ�ǰbox_map�Ƿ���Ϳ
if nargin < 2
    [h,w]=size(box_map);
    thresh = h*w*0.2;%Ĭ����ֵΪbox�������1/5
end
area = bwarea(box_map);
if area>thresh%���������ֵ
    opt=1;
else
    opt=0;
end
end