function rlt = get_rlt(opt_map)
%һ��opt_map��һ�����4��ѡ����ͼ
%�˺���ͨ��һ�����ѡ��ͼ�ж�ѡ����ǵڼ���ѡ��
%return 1,2,3,4����ABCD.
%5 ��ʾ����©��Ϳ�Ͷ���Ϳ���Ǵ��󣨵�ѡ��
[h,w]=size(opt_map);
dx = double(w)*0.25;
flag = 0;
rlt = 5;
for i=0:3
    opt = get_opt(opt_map(1:h,round(i*dx+1):floor((i+1)*dx)));
    if opt==1 
        if flag==0
            rlt = i+1;
            flag = 1;
        else
            rlt = 5;
        end
    end
end
end