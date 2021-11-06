function rlt = get_rlt(opt_map)
%一个opt_map是一道题的4个选项框的图
%此函数通过一道题的选项图判断选择的是第几个选项
%return 1,2,3,4代表ABCD.
%5 表示错误，漏填涂和多填涂都是错误（单选）
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